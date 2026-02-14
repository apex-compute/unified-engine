# Simple TCL script to program FPGA with bit file
# Usage: vivado -mode batch -nolog -nojournal -source program_fpga.tcl -tclargs <bit_file_path> [target_name]

# Suppress comment output and command echoing
set tcl_interactive 0
if {[info command set_echo] != ""} {
    set_echo off
}

# Expecting: <bit_file> [target_name]
if {[llength $argv] < 1} {
    puts "Error: Missing bit file argument"
    puts "Usage: vivado -mode batch -nolog -nojournal -source program_fpga.tcl -tclargs <bit_file_path> [target_name]"
    exit 1
}

set bit_file [lindex $argv 0]
set target_name ""
if {[llength $argv] >= 2} {
    set target_name [lindex $argv 1]
}

# Check if bit file exists
if {![file exists $bit_file]} {
    puts "Error: Bit file not found: $bit_file"
    exit 1
}

# Map the Makefile TARGET to the Silicon Part string
set part_map(kintex7) "xc7k480t"
set part_map(alveo) "xcu50_u55n"
set part_map(rk) "xcku5p"

puts "Opening hardware manager..."
open_hw_manager

puts "Connecting to hardware server..."
# Try to connect to existing server, if it fails, start server
if {[catch {connect_hw_server -url localhost:3121}]} {
    puts "Connection failed, starting hardware server..."
    start_hw_server
    connect_hw_server -url localhost:3121
}
puts "Connected to hardware server"

puts "Scanning all hardware targets for devices..."
set all_hw_devices {}
set device_to_target_map {}
set all_targets [get_hw_targets]

foreach target $all_targets {
    if {[catch {open_hw_target $target} err]} {
        puts "Warning: Could not open target $target: $err"
        continue
    }
    
    set devices [get_hw_devices]
    foreach device $devices {
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
    puts "Error: No hardware devices found"
    disconnect_hw_server
    close_hw_manager
    exit 1
}

set selected_device ""
set selected_target ""

if {$device_count == 1} {
    # Only one device, use it
    set selected_device [lindex $all_hw_devices 0]
    set selected_target [dict get $device_to_target_map $selected_device]
    open_hw_target $selected_target
    current_hw_device $selected_device
    puts "Using device: $selected_device (only one device available)"
} else {
    # Multiple devices, try to find matching one if target_name is specified
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
            # No match found, use the last device
            set selected_device [lindex $all_hw_devices end]
            set selected_target [dict get $device_to_target_map $selected_device]
            open_hw_target $selected_target
            current_hw_device $selected_device
            puts "No matching device found, using last device: $selected_device"
        }
    } else {
        # No target specified or target not in part_map, use the last device
        set selected_device [lindex $all_hw_devices end]
        set selected_target [dict get $device_to_target_map $selected_device]
        open_hw_target $selected_target
        current_hw_device $selected_device
        if {$target_name != ""} {
            puts "TARGET '$target_name' not in part_map, using last device: $selected_device"
        } else {
            puts "No TARGET specified, using last device: $selected_device"
        }
    }
}

# Print part info and expected target info
set actual_part [get_property PART $selected_device]
puts "Using device: $selected_device"
puts "Actual part: $actual_part"

if {$target_name != ""} {
    puts "Expected TARGET: $target_name"
    if {[info exists part_map($target_name)]} {
        puts "Expected part for TARGET=$target_name: $part_map($target_name)"
    } else {
        puts "Note: TARGET '$target_name' is not in part_map"
    }
}

# Program the found device
puts "Setting bit file: $bit_file"
set_property PROGRAM.FILE $bit_file [current_hw_device]

puts "Programming device..."
program_hw_devices [current_hw_device]

if {$target_name != ""} {
    puts "Programming $target_name successful!"
} else {
    puts "Programming successful!"
}

close_hw_target
disconnect_hw_server
close_hw_manager

puts "Done!"

# Clean up vivado journal and log files
set script_dir [file dirname [file normalize [info script]]]
foreach pattern {vivado*.jou vivado*.log} {
    foreach f [glob -nocomplain -directory $script_dir $pattern] {
        file delete -force $f
    }
}
