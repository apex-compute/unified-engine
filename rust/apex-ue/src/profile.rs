use std::fmt;

pub const EXPECTED_PUBLIC_HW_VERSION: u32 = 0x253d_5525;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DeviceProfile {
    pub name: &'static str,
    pub part: &'static str,
    pub family: &'static str,
    pub clock_ns: f64,
    pub axi_data_width_bits: u32,
    pub pcie_expectation: &'static str,
    pub expected_hw_version: Option<u32>,
}

impl DeviceProfile {
    pub fn peak_bf16_gflops(self) -> f64 {
        128.0 / self.clock_ns
    }
}

impl fmt::Display for DeviceProfile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}: {} / {}, {:.4} ns, {}b AXI, {:.1} BF16 GFLOPS peak, {}",
            self.name,
            self.part,
            self.family,
            self.clock_ns,
            self.axi_data_width_bits,
            self.peak_bf16_gflops(),
            self.pcie_expectation
        )
    }
}

pub const PROFILES: &[DeviceProfile] = &[
    DeviceProfile {
        name: "kintex7",
        part: "xc7k480t",
        family: "Kintex-7",
        clock_ns: 5.1594,
        axi_data_width_bits: 256,
        pcie_expectation:
            "Gen2-class endpoint, commonly up to x8; verify negotiated link with lspci.",
        expected_hw_version: Some(EXPECTED_PUBLIC_HW_VERSION),
    },
    DeviceProfile {
        name: "rk",
        part: "xcku5p",
        family: "Kintex UltraScale+",
        clock_ns: 3.0,
        axi_data_width_bits: 512,
        pcie_expectation: "Board-specific; verify negotiated link with lspci.",
        expected_hw_version: Some(EXPECTED_PUBLIC_HW_VERSION),
    },
    DeviceProfile {
        name: "puzhi",
        part: "xcku5p",
        family: "Kintex UltraScale+",
        clock_ns: 3.0,
        axi_data_width_bits: 256,
        pcie_expectation: "Board-specific; verify negotiated link with lspci.",
        expected_hw_version: Some(EXPECTED_PUBLIC_HW_VERSION),
    },
    DeviceProfile {
        name: "alveo",
        part: "xcu50/u55n",
        family: "Alveo UltraScale+",
        clock_ns: 4.0,
        axi_data_width_bits: 256,
        pcie_expectation: "Gen3 x16 or dual Gen4 x8 capable depending card and shell; not Gen5.",
        expected_hw_version: Some(EXPECTED_PUBLIC_HW_VERSION),
    },
    DeviceProfile {
        name: "bittware",
        part: "Kintex UltraScale KU15P class",
        family: "Kintex UltraScale+",
        clock_ns: 3.3333,
        axi_data_width_bits: 512,
        pcie_expectation: "Card-specific; verify negotiated link with lspci.",
        expected_hw_version: Some(EXPECTED_PUBLIC_HW_VERSION),
    },
    DeviceProfile {
        name: "bittware_256",
        part: "Kintex UltraScale KU15P class",
        family: "Kintex UltraScale+",
        clock_ns: 3.3333,
        axi_data_width_bits: 256,
        pcie_expectation: "Card-specific; verify negotiated link with lspci.",
        expected_hw_version: Some(EXPECTED_PUBLIC_HW_VERSION),
    },
];

pub fn all() -> &'static [DeviceProfile] {
    PROFILES
}

pub fn by_name(name: &str) -> Option<DeviceProfile> {
    PROFILES
        .iter()
        .copied()
        .find(|profile| profile.name.eq_ignore_ascii_case(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn looks_up_profiles_case_insensitively() {
        let profile = by_name("KINTEX7").expect("kintex7 profile");
        assert_eq!(profile.part, "xc7k480t");
        assert_eq!(
            profile.expected_hw_version,
            Some(EXPECTED_PUBLIC_HW_VERSION)
        );
    }

    #[test]
    fn peak_gflops_matches_readme_formula() {
        let profile = by_name("rk").expect("rk profile");
        assert!((profile.peak_bf16_gflops() - 42.666).abs() < 0.01);
    }
}
