"""Exercise provisioning decisions without connecting to or writing hardware."""

import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest

import provision_kintex7 as provision


ROOT = Path(__file__).resolve().parents[1]


def image(ctl=0x80000040, payload_words=4):
    words = [0xFFFFFFFF, 0xAA995566, 0x20000000,
             0x3000C001, 0x80000040, 0x3000A001, ctl,
             0x30016004, 1, 2, 3, 4, 0x30034001, payload_words,
             0, 0, 0, 0]
    data = bytearray(struct.pack(">" + "I" * len(words), *words).translate(provision.REVERSE_BITS))
    data[0::2], data[1::2] = data[1::2], data[0::2]
    return data


class Inputs(unittest.TestCase):
    def test_efuse_encrypted_bpi_image(self):
        provision.validate_image(image())

    def test_bbram_plain_and_truncated_images_rejected(self):
        for data in [image(0x40), image(0), image(payload_words=400), b"", b"wrong", image()[:-4]]:
            with self.subTest(data=data[:8]):
                with self.assertRaises(ValueError):
                    provision.validate_image(data)

    def test_key_device_format_and_secret_errors(self):
        key = "0123456789abcdef" * 4
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key.nky"
            path.write_text(f"Device xc7k480t;\nKey 0 {key};\n")
            self.assertEqual(provision.read_key(path), key)
            for content in [f"Device xcku5p; Key 0 {key};",
                            f"Device xc7k480t; Key 0 {key}; Key 0 {key};",
                            "Device xc7k480t; Key 0 " + "0" * 64 + ";"]:
                path.write_text(content)
                with self.assertRaises(ValueError) as error:
                    provision.read_key(path)
                self.assertNotIn(key, str(error.exception))

    def test_missing_sidecar_fails_before_vivado(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encrypted.bin"
            path.write_bytes(image())
            result = subprocess.run(["python3", str(ROOT / "provision_kintex7.py"), str(path),
                                     "--program", "--vivado", "/nonexistent/vivado"],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Matching NKY not found", result.stderr)


# Handles contain a generation number. Reusing a handle after close/open fails,
# and both cables may expose the exact same device name. This catches the
# cross-cable aliasing bug of indexing a device->cable map by device name alone.
MOCK = r'''
set scenario $::env(SCENARIO)
set opened ""
set epoch 0
set aes [expr {$scenario in {existing flash_failure}}]
set control [expr {$scenario eq "locked" ? 12 : $scenario eq "aes_only" ? 1 : 192}]
set connects 0
proc open_hw_manager {} {}
proc connect_hw_server {args} {if {$::scenario eq "server_failure"} {error "server unavailable"}}
proc disconnect_hw_server {} {}
proc close_hw_manager {} {}
proc get_hw_targets {} {return {server/Digilent/A server/Digilent/B}}
proc current_hw_target {{target ""}} {
    if {$target ne ""} {set ::current $target}
    return $::current
}
proc open_hw_target {target} {
    if {$::scenario eq "scan_failure" && [file tail $target] eq "B"} {error "cable busy"}
    set ::opened $target
    incr ::epoch
    if {[file tail $target] eq "B"} {incr ::connects}
}
proc close_hw_target {} {set ::opened ""}
proc get_hw_devices {args} {
    if {$args ne [list -of_objects $::current]} {error "Devices must be scoped to current cable"}
    set cable [file tail $::opened]
    set part [expr {$cable eq "A" && $::scenario ni {ambiguous select_a} ? "xcku5p" : "xc7k480t"}]
    if {$::scenario eq "no_match"} {set part xcku5p}
    return [list "$::epoch|$cable|${part}_0"]
}
proc check_handle {object} {
    lassign [split $object |] epoch cable name
    if {$epoch != $::epoch || $cable ne [file tail $::opened]} {error "STALE/CROSS-CABLE HANDLE"}
    return $name
}
proc get_property {property object} {
    if {[string match server/* $object]} {return $object}
    set name [check_handle $object]
    switch -- $property {
        NAME {return $name}
        PART {
            if {$::scenario eq "changed_part" && $::connects > 1} {return xcku5p}
            return [lindex [split $name _] 0]
        }
        REGISTER.EFUSE.FUSE_DNA {
            if {$::scenario eq "changed_dna" && $::connects > 1} {return ABC12399}
            return [expr {[file tail $::opened] eq "A" ? "ABC12301" : "ABC12302"}]
        }
        PROGRAM.IS_AES_PROGRAMMED {
            if {$::scenario eq "unknown_aes"} {return ""}
            return $::aes
        }
        REGISTER.EFUSE.FUSE_CNTL {return [format %X $::control]}
        PROGRAM.HW_CFGMEM {return mock_cfgmem}
        PROGRAM.HW_CFGMEM_BITFILE {return helper.bit}
        REGISTER.CONFIG_STATUS.BIT14_DONE_PIN {return [expr {$::scenario ne "boot_failure"}]}
        default {error "Unexpected property $property"}
    }
}
proc current_hw_device {dev} {check_handle $dev}
proc refresh_hw_device {args} {check_handle [lindex $args end]}
proc get_cfgmem_parts {args} {
    if {[lindex $args 0] ne "-of_objects"} {error "Flash part query must be scoped to FPGA"}
    if {$::scenario eq "no_cfgmem"} {return {}}
    return mt28gu512aax1e-bpi-x16
}
proc create_hw_cfgmem {args} {puts "PREPARE cfgmem"}
proc create_hw_bitstream {args} {puts "PREPARE bitstream"}
proc set_property {args} {}
proc program_hw_devices {args} {
    set dev [lindex $args end]
    check_handle $dev
    if {[lsearch -exact $args -key] >= 0} {
        if {[lsearch -exact $args -force] >= 0} {error "force not permitted"}
        puts "WRITE burn [file tail $::opened]"
        if {$::scenario eq "burn_failure"} {error "burn failed"}
        if {$::scenario ne "false_burn_status"} {set ::aes 1}
        set ::control 204
        set path [lindex $args [expr {[lsearch -exact $args -efuse_export_file] + 1}]]
        close [open $path w]
    } else {puts "WRITE helper [file tail $::opened]"}
}
proc program_hw_cfgmem {args} {
    puts "WRITE flash [file tail $::opened]"
    if {$::scenario eq "flash_failure"} {error "flash verify failed"}
}
proc boot_hw_device {dev} {check_handle $dev; puts "WRITE boot [file tail $::opened]"}
proc after {args} {}
source $::env(BACKEND)
'''


@unittest.skipUnless(shutil.which("tclsh"), "tclsh required for hardware mocks")
class HardwareFlow(unittest.TestCase):
    def run_flow(self, scenario="normal", mode="program", **options):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "mock.tcl").write_text(MOCK)
            (path / "image.bin").write_bytes(image())
            (path / "key.nky").write_text("synthetic fixture")
            env = dict(os.environ, SCENARIO=scenario, BACKEND=str(ROOT / "provision_kintex7.tcl"),
                       K7_MODE=mode, K7_TARGET="", K7_DEVICE="", K7_DNA="", K7_SERVER="localhost:3121", K7_BOOT="0")
            env.update({f"K7_{key}": value for key, value in options.items()})
            result = subprocess.run(["tclsh", "mock.tcl"], cwd=path, env=env,
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual((path / "SUCCESS").exists(), result.returncode == 0, result.stdout + result.stderr)
            return result, [line for line in result.stdout.splitlines() if line.startswith("WRITE ")]

    def test_selects_kintex_on_second_cable_and_programs_in_order(self):
        result, writes = self.run_flow(BOOT="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(writes, ["WRITE burn B", "WRITE helper B", "WRITE flash B", "WRITE boot B"])

    def test_read_only_modes_do_not_prepare_or_program(self):
        for mode in ["list", "check"]:
            result, writes = self.run_flow(mode=mode)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(writes)
            self.assertNotIn("PREPARE", result.stdout)

    def test_failures_before_writes(self):
        for scenario in ["server_failure", "ambiguous", "scan_failure", "no_match",
                         "changed_dna", "changed_part", "existing", "locked", "aes_only",
                         "unknown_aes", "no_cfgmem"]:
            with self.subTest(scenario=scenario):
                result, writes = self.run_flow(scenario)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertFalse(writes, result.stdout)

    def test_same_named_devices_can_be_selected_by_cable_or_dna(self):
        for selectors in [dict(TARGET="A"), dict(DNA="0xABC12301")]:
            result, writes = self.run_flow("select_a", **selectors)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(writes, ["WRITE burn A", "WRITE helper A", "WRITE flash A"])

    def test_wrong_selectors_do_not_write(self):
        for selectors in [dict(TARGET="missing"), dict(DEVICE="xc7k480t_1"), dict(DNA="123")]:
            result, writes = self.run_flow(**selectors)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(writes)

    def test_failed_burn_never_flashes(self):
        for scenario in ["burn_failure", "false_burn_status"]:
            result, writes = self.run_flow(scenario)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(writes, ["WRITE burn B"])

    def test_flash_only_never_burns_and_requires_existing_key(self):
        result, writes = self.run_flow("existing", mode="flash")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(writes, ["WRITE helper B", "WRITE flash B"])
        result, writes = self.run_flow(mode="flash")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(writes)

    def test_verify_failure_never_boots(self):
        result, writes = self.run_flow("flash_failure", mode="flash", BOOT="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("WRITE boot B", writes)

    def test_failed_boot_reported_as_failure(self):
        result, writes = self.run_flow("boot_failure", BOOT="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("WRITE boot B", writes)


if __name__ == "__main__":
    unittest.main()
