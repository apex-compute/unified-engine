# Simple TCL script to boot FPGA from flash

# Suppress comment output and command echoing
set tcl_interactive 0
if {[info command set_echo] != ""} {
    set_echo off
}

open_hw_manager
connect_hw_server -allow_non_jtag

# Map the Makefile TARGET to the Silicon Part string
set target_name ""
if {[llength $argv] >= 1} {
    set target_name [lindex $argv 0]
}

set part_map(kintex7) "xc7k480t"
set part_map(kintex7_closed_timing) "xc7k480t"
set part_map(kintex7_systolic) "xc7k480t"
set part_map(alveo) "xcu50_u55n"
set part_map(rk) "xcku5p"
set part_map(puzhi) "xcku5p"

# Enumerate devices on every hw_target (same approach as program_fpga.tcl).
# A bare open_hw_target only attaches to the default target and misses devices
# on other cables / scan chains.
puts "Scanning all hardware targets for devices..."
set all_hw_devices {}
set device_to_target_map {}
set all_targets [get_hw_targets]

foreach target $all_targets {
    if {[catch {open_hw_target $target} err]} {
        puts "Warning: Could not open target $target: $err"
        continue
    }
    foreach device [get_hw_devices] {
        lappend all_hw_devices $device
        dict set device_to_target_map $device $target
    }
    close_hw_target
}

set device_count [llength $all_hw_devices]
puts "Found $device_count device(s) across all targets:"
foreach device $all_hw_devices {
    puts "  - $device"
}

if {$device_count == 0} {
    puts "Error: No hardware devices found."
    disconnect_hw_server
    close_hw_manager
    exit 1
}

set selected_device ""
set selected_target ""

if {$device_count == 1} {
    set selected_device [lindex $all_hw_devices 0]
    set selected_target [dict get $device_to_target_map $selected_device]
    if {[catch {open_hw_target $selected_target} err]} {
        puts "Error: Could not open hardware target $selected_target: $err"
        disconnect_hw_server
        close_hw_manager
        exit 1
    }
    current_hw_device $selected_device
    puts "Using device: $selected_device (only one device available)"
} else {
    if {$target_name != "" && [info exists part_map($target_name)]} {
        set search_part $part_map($target_name)
        puts "Multiple devices found, searching for matching part: $search_part (TARGET=$target_name)..."

        set found 0
        foreach device $all_hw_devices {
            set target [dict get $device_to_target_map $device]
            if {[catch {open_hw_target $target} err]} { continue }
            set actual_part [get_property PART $device]
            if {[string match "*$search_part*" $actual_part]} {
                set selected_device $device
                set selected_target $target
                current_hw_device $device
                puts "MATCH FOUND: Using device $device with part $actual_part"
                set found 1
                break
            }
            close_hw_target
        }

        if {!$found} {
            set selected_device [lindex $all_hw_devices end]
            set selected_target [dict get $device_to_target_map $selected_device]
            if {[catch {open_hw_target $selected_target} err]} {
                puts "Error: Could not open hardware target $selected_target: $err"
                disconnect_hw_server
                close_hw_manager
                exit 1
            }
            current_hw_device $selected_device
            puts "No matching device found, using last device: $selected_device"
        }
    } else {
        set selected_device [lindex $all_hw_devices end]
        set selected_target [dict get $device_to_target_map $selected_device]
        if {[catch {open_hw_target $selected_target} err]} {
            puts "Error: Could not open hardware target $selected_target: $err"
            disconnect_hw_server
            close_hw_manager
            exit 1
        }
        current_hw_device $selected_device
        if {$target_name != ""} {
            puts "TARGET '$target_name' not in part_map, using last device: $selected_device"
        } else {
            puts "No TARGET specified, using last device: $selected_device"
        }
    }
}

refresh_hw_device -update_hw_probes false $selected_device
boot_hw_device $selected_device

close_hw_target
disconnect_hw_server
close_hw_manager

# Clean up vivado journal and log files
set script_dir [file dirname [file normalize [info script]]]
foreach pattern {vivado*.jou vivado*.log} {
    foreach f [glob -nocomplain -directory $script_dir $pattern] {
        file delete -force $f
    }
}
