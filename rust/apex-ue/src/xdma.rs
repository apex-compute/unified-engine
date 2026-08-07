use crate::profile::DeviceProfile;
use std::fmt;
use std::fs::{self, File};
use std::io;
use std::path::PathBuf;

#[cfg(unix)]
use std::os::unix::fs::FileExt;

pub const AXI_LITE_TRANSLATION_OFFSET: u64 = 0x0000_0000;
pub const UE_0_BASE_ADDR: u64 = 0x0200_0000;
pub const UE_FPGA_VERSION_ADDR: u64 = 0x0000_0000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeStatus {
    Ready,
    MissingDevice,
    PermissionDenied,
    ReadError,
    VersionMismatch,
    NoExpectedVersion,
}

impl fmt::Display for ProbeStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Ready => "ready",
            Self::MissingDevice => "missing-device",
            Self::PermissionDenied => "permission-denied",
            Self::ReadError => "read-error",
            Self::VersionMismatch => "version-mismatch",
            Self::NoExpectedVersion => "no-expected-version",
        };
        f.write_str(label)
    }
}

#[derive(Debug, Clone)]
pub struct XdmaDevice {
    pub name: String,
    pub base_addr: u64,
}

impl XdmaDevice {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            base_addr: UE_0_BASE_ADDR,
        }
    }

    pub fn with_base_addr(mut self, base_addr: u64) -> Self {
        self.base_addr = base_addr;
        self
    }

    pub fn user_path(&self) -> PathBuf {
        PathBuf::from(format!("/dev/{}_user", self.name))
    }

    pub fn h2c_path(&self) -> PathBuf {
        PathBuf::from(format!("/dev/{}_h2c_0", self.name))
    }

    pub fn c2h_path(&self) -> PathBuf {
        PathBuf::from(format!("/dev/{}_c2h_0", self.name))
    }

    pub fn read_user_reg32(&self, reg_offset: u64) -> io::Result<u32> {
        let file = File::open(self.user_path())?;
        let offset = self.base_addr + reg_offset - AXI_LITE_TRANSLATION_OFFSET;
        let mut bytes = [0_u8; 4];
        #[cfg(unix)]
        {
            file.read_at(&mut bytes, offset)?;
        }
        #[cfg(not(unix))]
        {
            let _ = file;
            let _ = offset;
            return Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "XDMA register reads require a Unix FileExt::read_at implementation",
            ));
        }
        Ok(u32::from_le_bytes(bytes))
    }

    pub fn probe(&self, profile: DeviceProfile) -> ProbeReport {
        let mut messages = Vec::new();
        let expected_hw_version = profile.expected_hw_version;
        let required_paths = [self.user_path(), self.h2c_path(), self.c2h_path()];
        let mut missing = Vec::new();

        for path in &required_paths {
            if let Err(err) = fs::metadata(path) {
                if err.kind() == io::ErrorKind::NotFound {
                    missing.push(path.display().to_string());
                } else if err.kind() == io::ErrorKind::PermissionDenied {
                    messages.push(format!(
                        "permission denied reading metadata for {}",
                        path.display()
                    ));
                    return ProbeReport {
                        device: self.name.clone(),
                        profile,
                        expected_hw_version,
                        observed_hw_version: None,
                        status: ProbeStatus::PermissionDenied,
                        messages,
                    };
                } else {
                    messages.push(format!("could not inspect {}: {}", path.display(), err));
                    return ProbeReport {
                        device: self.name.clone(),
                        profile,
                        expected_hw_version,
                        observed_hw_version: None,
                        status: ProbeStatus::ReadError,
                        messages,
                    };
                }
            }
        }

        if !missing.is_empty() {
            messages.push(format!("missing XDMA nodes: {}", missing.join(", ")));
            messages.push(
                "load the Xilinx XDMA driver and confirm the card enumerated on this host"
                    .to_string(),
            );
            return ProbeReport {
                device: self.name.clone(),
                profile,
                expected_hw_version,
                observed_hw_version: None,
                status: ProbeStatus::MissingDevice,
                messages,
            };
        }

        let observed_hw_version = match self.read_user_reg32(UE_FPGA_VERSION_ADDR) {
            Ok(value) => Some(value),
            Err(err) if err.kind() == io::ErrorKind::PermissionDenied => {
                messages.push(format!(
                    "permission denied opening {}",
                    self.user_path().display()
                ));
                return ProbeReport {
                    device: self.name.clone(),
                    profile,
                    expected_hw_version,
                    observed_hw_version: None,
                    status: ProbeStatus::PermissionDenied,
                    messages,
                };
            }
            Err(err) => {
                messages.push(format!("could not read UE version register: {}", err));
                return ProbeReport {
                    device: self.name.clone(),
                    profile,
                    expected_hw_version,
                    observed_hw_version: None,
                    status: ProbeStatus::ReadError,
                    messages,
                };
            }
        };

        match expected_hw_version {
            Some(expected) if observed_hw_version == Some(expected) => {
                messages.push(format!(
                    "hardware version matched expected public bitstream 0x{expected:08x}"
                ));
                ProbeReport {
                    device: self.name.clone(),
                    profile,
                    expected_hw_version,
                    observed_hw_version,
                    status: ProbeStatus::Ready,
                    messages,
                }
            }
            Some(expected) => {
                let observed = observed_hw_version.unwrap_or_default();
                messages.push(format!(
                    "hardware version mismatch: observed 0x{observed:08x}, expected 0x{expected:08x}"
                ));
                ProbeReport {
                    device: self.name.clone(),
                    profile,
                    expected_hw_version,
                    observed_hw_version,
                    status: ProbeStatus::VersionMismatch,
                    messages,
                }
            }
            None => {
                messages.push(
                    "profile has no expected hardware version; refusing to launch inference"
                        .to_string(),
                );
                ProbeReport {
                    device: self.name.clone(),
                    profile,
                    expected_hw_version,
                    observed_hw_version,
                    status: ProbeStatus::NoExpectedVersion,
                    messages,
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProbeReport {
    pub device: String,
    pub profile: DeviceProfile,
    pub expected_hw_version: Option<u32>,
    pub observed_hw_version: Option<u32>,
    pub status: ProbeStatus,
    pub messages: Vec<String>,
}

impl ProbeReport {
    pub fn is_ready(&self) -> bool {
        self.status == ProbeStatus::Ready
    }
}
