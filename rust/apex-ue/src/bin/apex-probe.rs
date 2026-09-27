use apex_ue::profile;
use apex_ue::xdma::{ProbeStatus, XdmaDevice, UE_0_BASE_ADDR};
use std::env;

fn main() {
    let mut dev = "xdma0".to_string();
    let mut profile_name = "kintex7".to_string();
    let mut base_addr = UE_0_BASE_ADDR;
    let mut show_profiles = false;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dev" => dev = require_value("--dev", args.next()),
            "--profile" => profile_name = require_value("--profile", args.next()),
            "--base" => {
                let raw = require_value("--base", args.next());
                base_addr = parse_u64(&raw).unwrap_or_else(|err| fail(&err));
            }
            "--profiles" => show_profiles = true,
            "-h" | "--help" => {
                usage();
                return;
            }
            other => fail(&format!("unknown argument: {other}")),
        }
    }

    if show_profiles {
        for item in profile::all() {
            println!("{item}");
        }
        return;
    }

    let selected = profile::by_name(&profile_name).unwrap_or_else(|| {
        fail(&format!(
            "unknown profile {profile_name:?}; pass --profiles to list supported boards"
        ))
    });

    let device = XdmaDevice::new(dev).with_base_addr(base_addr);
    let report = device.probe(selected);

    println!("Apex Unified Engine probe");
    println!("  device   : {}", report.device);
    println!("  profile  : {}", report.profile.name);
    println!("  part     : {}", report.profile.part);
    println!("  status   : {}", report.status);
    println!(
        "  expected : {}",
        report
            .expected_hw_version
            .map(|v| format!("0x{v:08x}"))
            .unwrap_or_else(|| "none".to_string())
    );
    println!(
        "  observed : {}",
        report
            .observed_hw_version
            .map(|v| format!("0x{v:08x}"))
            .unwrap_or_else(|| "not-read".to_string())
    );
    for message in &report.messages {
        println!("  - {message}");
    }

    if report.status != ProbeStatus::Ready {
        std::process::exit(2);
    }
}

fn usage() {
    println!("usage: apex-probe [--dev xdma0] [--profile kintex7] [--base 0x02000000]");
    println!("       apex-probe --profiles");
}

fn require_value(flag: &str, value: Option<String>) -> String {
    value.unwrap_or_else(|| fail(&format!("{flag} requires a value")))
}

fn parse_u64(raw: &str) -> Result<u64, String> {
    if let Some(hex) = raw.strip_prefix("0x").or_else(|| raw.strip_prefix("0X")) {
        u64::from_str_radix(hex, 16).map_err(|err| err.to_string())
    } else {
        raw.parse::<u64>().map_err(|err| err.to_string())
    }
}

fn fail(message: &str) -> ! {
    eprintln!("error: {message}");
    std::process::exit(1);
}
