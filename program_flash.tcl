# Program configuration flash PROM with MCS file
# Usage: vivado -mode batch -source program_flash.tcl -tclargs <target> [BITFILE=<path>]
#        or: make program_flash TARGET=<target> [BITFILE=<path>]
#        TARGET: alveo, puzhi, alinx, or kintex7 (default: alveo)
#        BITFILE: optional path to bitstream; if omitted, default path is used

# Parse target and optional BITFILE= from command line
set target "alveo"
set custom_bitfile ""
foreach a $argv {
    if { [string match "BITFILE=*" $a] } {
        set custom_bitfile [file normalize [string range $a 8 end]]
    } else {
        set target $a
    }
}

puts "Target: $target"

# Get script directory and set paths
set SCRIPT_DIR [file dirname [file normalize [info script]]]
set MCS_FILE [file join $SCRIPT_DIR "${target}.mcs"]

# Bitstream path: use BITFILE= if provided, else default path
# Default: andromeda/Vivado/{TARGET}/{TARGET}.runs/impl_1/andromeda_wrapper.bit
if { $custom_bitfile != "" } {
    set BITSTREAM_FILE $custom_bitfile
    puts "Using custom bitfile: $BITSTREAM_FILE"
} else {
    set BITSTREAM_FILE [file join $SCRIPT_DIR ".." "Vivado" $target "${target}.runs" "impl_1" "andromeda_wrapper.bit"]
    set BITSTREAM_FILE [file normalize $BITSTREAM_FILE]
}

# Require bitstream; generate MCS from it (always overwrite)
if { ![file exists $BITSTREAM_FILE] } {
    puts "ERROR: Bitstream not found at: $BITSTREAM_FILE"
    puts "Please run build first or pass BITFILE=<path>."
    error "Bitstream not found"
}

puts "Generating MCS from bitstream: $BITSTREAM_FILE -> $MCS_FILE"
if { $target == "alveo" } {
    write_cfgmem -force -format mcs -interface spix4 -size 128 \
        -loadbit "up 0x01002000 $BITSTREAM_FILE" \
        -file $MCS_FILE
} elseif { $target == "kintex7" } {
    write_cfgmem -force -format mcs -size 64 -interface BPIx16 \
        -loadbit "up 0x00000000 $BITSTREAM_FILE" \
        -file $MCS_FILE
} else {
    puts "Target '$target' is not supported. Add MCS config in program_flash.tcl."
    error "Unsupported target: $target"
}
if { ![file exists $MCS_FILE] } {
    puts "ERROR: MCS generation failed"
    error "MCS file not created"
}
puts "MCS generated: $MCS_FILE"

# Open hardware manager (required for hardware manager commands)
puts "Opening hardware manager..."
open_hw_manager

# Connect to hardware server (local or remote)
puts "Connecting to hardware server..."
if { [catch {connect_hw_server -allow_non_jtag -url localhost:3121} result] } {
    puts "Warning: Could not connect to existing hardware server, starting new one..."
    start_hw_server
    connect_hw_server -allow_non_jtag
}

# Map TARGET to silicon PART string (same as program_fpga.tcl) for hardware verification
set part_map(kintex7) "xc7k480t"
set part_map(alveo)   "xcu50_u55n"

# Scan all hardware targets and collect devices + which target each device is on
puts "Scanning all hardware targets for devices..."
set all_hw_devices {}
set device_to_target_map {}
foreach hw_tgt [get_hw_targets] {
    if { [catch {open_hw_target $hw_tgt} err] } {
        puts "Warning: Could not open target $hw_tgt: $err"
        continue
    }
    foreach dev [get_hw_devices] {
        lappend all_hw_devices $dev
        dict set device_to_target_map $dev $hw_tgt
    }
    close_hw_target
}

set device_count [llength $all_hw_devices]
if { $device_count == 0 } {
    puts "ERROR: No hardware devices found."
    puts "Please ensure the board is connected, JTAG is connected, and power is on."
    close_hw_manager
    error "No hardware devices found"
}
puts "Found $device_count device(s)"

# Select device by matching PART to expected part for this target
set selected_device ""
set selected_target ""
if { ![info exists part_map($target)] } {
    puts "Target '$target' is not supported for flash programming."
    puts "Manually add the program flash settings for this target in program_flash.tcl."
    close_hw_manager
    error "Unsupported target: $target"
}
set search_part $part_map($target)
puts "Selecting device for target $target (expected part pattern: *$search_part*)..."

set found 0
foreach dev $all_hw_devices {
    set hw_tgt [dict get $device_to_target_map $dev]
    if { [catch {open_hw_target $hw_tgt} err] } { continue }
    set actual_part [get_property PART $dev]
    set match 0
    if { $target == "alveo" } {
        if { [string match "*xcu50*" $actual_part] || [string match "*u55n*" $actual_part] } { set match 1 }
    } else {
        if { [string match "*${search_part}*" $actual_part] } { set match 1 }
    }
    if { $match } {
        set selected_device $dev
        set selected_target $hw_tgt
        puts "MATCH: Using device $dev (PART $actual_part)"
        set found 1
        break
    }
    close_hw_target
}

if { !$found } {
    set selected_device [lindex $all_hw_devices 0]
    set selected_target [dict get $device_to_target_map $selected_device]
    open_hw_target $selected_target
    set actual_part [get_property PART $selected_device]
    puts "No matching part found; using first device: $selected_device (PART $actual_part)"
    puts "Expected for TARGET=$target: *$search_part*"
} else {
    open_hw_target $selected_target
}
current_hw_device $selected_device
refresh_hw_device -update_hw_probes false $selected_device
set device $selected_device

# Create flash configuration memory device (target-specific part and options)
set hw_device [lindex [get_hw_devices $device] 0]
if { $target == "alveo" } {
    puts "Creating flash config (Alveo: mt25qu01g SPI)..."
    create_hw_cfgmem -hw_device $hw_device -mem_dev [lindex [get_cfgmem_parts {mt25qu01g-spi-x1_x2_x4}] 0]
} elseif { $target == "kintex7" } {
    puts "Creating flash config (Kintex-7: mt28gu512aax1e BPI-x16)..."
    create_hw_cfgmem -hw_device $hw_device -mem_dev [lindex [get_cfgmem_parts {mt28gu512aax1e-bpi-x16}] 0]
} else {
    puts "Target '$target' is not supported for flash programming."
    puts "Manually add the program flash settings for this target in program_flash.tcl."
    close_hw_target
    error "Unsupported target for flash programming: $target"
}

set cfgmem [get_property PROGRAM.HW_CFGMEM $hw_device]
set_property PROGRAM.BLANK_CHECK  0 $cfgmem
set_property PROGRAM.ERASE        1 $cfgmem
set_property PROGRAM.CFG_PROGRAM  1 $cfgmem
set_property PROGRAM.CHECKSUM     0 $cfgmem
set_property PROGRAM.ADDRESS_RANGE {use_file} $cfgmem
set_property PROGRAM.UNUSED_PIN_TERMINATION {pull-none} $cfgmem
set_property PROGRAM.VERIFY  1 $cfgmem

# Set the MCS file to program
puts "Setting MCS file: $MCS_FILE"
set_property PROGRAM.FILES [list $MCS_FILE] $cfgmem
set_property PROGRAM.PRM_FILE {} $cfgmem

# Program device with bitstream first (sets up interface for flash), then program flash
puts "Programming device with bitstream from MCS file..."
create_hw_bitstream -hw_device $hw_device [get_property PROGRAM.HW_CFGMEM_BITFILE $hw_device]
program_hw_devices $hw_device
refresh_hw_device $hw_device

puts "Programming flash memory with MCS file..."
program_hw_cfgmem -hw_cfgmem $cfgmem

puts "Flash programming complete!"

# Close hardware target
close_hw_target