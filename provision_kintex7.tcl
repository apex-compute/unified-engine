# Backend for provision_kintex7.py. Run the Python entry point so that the BIN
# and NKY are validated and Vivado runs in a private working directory.
set tcl_interactive 0
if {[info commands set_echo] ne ""} {set_echo off}

proc shutdown {} {
    catch {close_hw_target}
    catch {disconnect_hw_server}
    catch {close_hw_manager}
}

proc hex_value {value} {
    set value [string map {_ ""} [string trim $value]]
    regsub -nocase {^0x} $value "" value
    if {![regexp {^[0-9a-fA-F]+$} $value]} {error "Invalid/missing hexadecimal hardware property"}
    return [expr "0x$value"]
}

proc aes_programmed {dev} {
    set value [get_property PROGRAM.IS_AES_PROGRAMMED $dev]
    if {![string is boolean -strict $value]} {error "Unknown AES eFUSE status; refusing to proceed"}
    return [expr {$value ? 1 : 0}]
}

proc identity {dev} {
    refresh_hw_device -update_hw_probes false $dev
    set dna [hex_value [get_property REGISTER.EFUSE.FUSE_DNA $dev]]
    if {$dna == 0} {error "Missing device DNA; refusing to proceed"}
    return [format %016X $dna]
}

proc run {} {
    foreach name {MODE TARGET DEVICE DNA SERVER BOOT} {
        if {![info exists ::env(K7_$name)]} {error "Use provision_kintex7.py"}
        set opt($name) $::env(K7_$name)
    }
    if {$opt(MODE) ni {list check program flash}} {error "Invalid mode"}
    if {$opt(MODE) ne "list" && ![file isfile image.bin]} {error "Missing validated BIN snapshot"}
    if {$opt(MODE) in {check program} && ![file isfile key.nky]} {error "Missing validated NKY snapshot"}

    open_hw_manager
    # connect_hw_server starts the local server if necessary. Never silently
    # fall back to a different server when an explicitly supplied URL fails.
    connect_hw_server -url $opt(SERVER)
    set targets {}
    foreach target [get_hw_targets] {
        set name [get_property NAME $target]
        if {$opt(TARGET) eq "" || $name eq $opt(TARGET) || [file tail $name] eq $opt(TARGET)} {
            lappend targets $target
        }
    }
    if {[llength $targets] == 0} {error "No matching JTAG cables found"}
    if {$opt(TARGET) ne "" && [llength $targets] != 1} {error "Cable selector is not unique; use its full target path"}

    set candidates {}
    set failures {}
    foreach target $targets {
        set target_name [get_property NAME $target]
        puts "Cable: $target_name"
        if {[catch {
            current_hw_target $target
            open_hw_target $target
            foreach dev [get_hw_devices -of_objects [current_hw_target]] {
                set name [get_property NAME $dev]
                set part [string tolower [get_property PART $dev]]
                puts "  Device: $name  PART=$part"
                if {$part ne "xc7k480t"} {continue}
                if {$opt(DEVICE) ne "" && $name ne $opt(DEVICE)} {continue}
                set dna [identity $dev]
                puts "    DNA=$dna  AES_PROGRAMMED=[aes_programmed $dev]"
                if {$opt(DNA) ne "" && [hex_value $opt(DNA)] != [hex_value $dna]} {continue}
                # Store strings, never handles belonging to a closed target.
                lappend candidates [list $target_name $name $part $dna]
            }
        } message]} {
            lappend failures $target_name
            puts "  SCAN FAILED: $message"
        }
        catch {close_hw_target}
    }
    if {[llength $failures]} {
        error "Incomplete JTAG scan; select a reachable cable with --target before retrying"
    }
    if {$opt(MODE) eq "list"} {return}
    if {[llength $candidates] != 1} {
        error "Expected one XC7K480T, found [llength $candidates]; use --target, --device and/or --dna"
    }
    lassign [lindex $candidates 0] target_name device_name part dna

    # Reacquire both objects after reopening, scoped to this cable. Identical
    # names (xc7k480t_0) on other cables must not alias this device.
    set selected_target ""
    foreach target [get_hw_targets] {
        if {[get_property NAME $target] eq $target_name} {set selected_target $target}
    }
    if {$selected_target eq ""} {error "Selected cable disappeared"}
    current_hw_target $selected_target
    open_hw_target $selected_target
    set matches {}
    foreach dev [get_hw_devices -of_objects [current_hw_target]] {
        if {[get_property NAME $dev] eq $device_name} {lappend matches $dev}
    }
    if {[llength $matches] != 1} {error "Selected device disappeared or is ambiguous"}
    set dev [lindex $matches 0]
    current_hw_device $dev
    if {[string tolower [get_property PART $dev]] ne $part || [identity $dev] ne $dna} {
        error "Selected device identity changed during scan"
    }
    set aes [aes_programmed $dev]
    set control [hex_value [get_property REGISTER.EFUSE.FUSE_CNTL $dev]]
    puts "Selected: $target_name / $device_name  PART=$part  DNA=$dna"
    puts [format "AES_PROGRAMMED=%d  FUSE_CNTL=0x%04X" $aes $control]
    if {$control & 1} {error "CFG_AES_Only is set; Vivado indirect BPI flash programming is unavailable"}
    if {$opt(MODE) eq "program" && $aes} {
        error "AES is already fused; refusing to reburn. Use --flash-only with an image for the existing key"
    }
    if {$opt(MODE) in {check program} && !$aes && ($control & 0x3C)} {
        error "eFUSE key/user/control access is locked; refusing to program"
    }
    if {$opt(MODE) eq "flash" && !$aes} {error "--flash-only requires an already-programmed AES eFUSE key"}

    # Unscoped Vivado 2024.2 queries return this name once per FPGA family.
    set parts [get_cfgmem_parts -of_objects $dev -filter {NAME == "mt28gu512aax1e-bpi-x16"}]
    if {[llength $parts] != 1} {error "Required MT28GU512 BPI-x16 flash part is unavailable"}
    set record [open device.txt w]
    puts $record "TARGET=$target_name\nDEVICE=$device_name\nPART=$part\nDNA=$dna\nAES_PROGRAMMED=$aes"
    close $record
    if {$opt(MODE) eq "check"} {
        puts "CHECK COMPLETE: no device programming, flash erase or eFUSE burn performed."
        if {$aes} {puts "Device already fused; --program will refuse. Use --flash-only for updates."}
        puts "Provisioning policy: FUSE_USER=0, FUSE_CNTL=0x0c; CFG_AES_Only stays unset."
        return
    }

    # Resolve all flash settings and create its helper before touching OTP.
    create_hw_cfgmem -hw_device $dev -mem_dev [lindex $parts 0]
    set cfgmem [get_property PROGRAM.HW_CFGMEM $dev]
    foreach {property value} {
        PROGRAM.BLANK_CHECK 0 PROGRAM.ERASE 1 PROGRAM.CFG_PROGRAM 1
        PROGRAM.CHECKSUM 0 PROGRAM.ADDRESS_RANGE use_file
        PROGRAM.UNUSED_PIN_TERMINATION pull-none PROGRAM.VERIFY 1
    } {set_property $property $value $cfgmem}
    set_property PROGRAM.FILES [list [file normalize image.bin]] $cfgmem
    set_property PROGRAM.PRM_FILE {} $cfgmem
    set helper [get_property PROGRAM.HW_CFGMEM_BITFILE $dev]
    create_hw_bitstream -hw_device $dev $helper

    if {$opt(MODE) eq "program"} {
        create_hw_bitstream -hw_device $dev -nky [file normalize key.nky]
        # Check again immediately before the one-time operation. Do not force
        # retries or suppress Vivado programming errors with -quiet.
        if {[identity $dev] ne $dna || [aes_programmed $dev]} {error "Device changed before eFUSE burn"}
        puts "BURNING AES eFUSE and key read/write protection (0x0c); this is permanent."
        puts "FUSE_USER=0; programming AES also consumes the USER low-byte provisioning opportunity."
        program_hw_devices -key efuse -skip_program_rsa -user_efuse 0 -control_efuse 0c \
            -efuse_export_file [file normalize "efuse_${dna}.nkz"] $dev
        refresh_hw_device -update_hw_probes false $dev
        if {![aes_programmed $dev]} {error "AES programmed status was not asserted after burn"}
        set control [hex_value [get_property REGISTER.EFUSE.FUSE_CNTL $dev]]
        if {($control & 0x0C) != 0x0C || ($control & 1)} {error "Unexpected FUSE_CNTL after burn"}
        puts "eFUSE programming completed. NKZ export: [file normalize efuse_${dna}.nkz]"
    }

    # The key-only hw_bitstream replaces the helper association; restore it.
    puts "Loading BPI flash programming helper on $device_name..."
    create_hw_bitstream -hw_device $dev $helper
    program_hw_devices $dev
    refresh_hw_device -update_hw_probes false $dev
    puts "Erasing, programming and verifying BPI flash..."
    program_hw_cfgmem -hw_cfgmem $cfgmem
    puts "FLASH VERIFIED."
    if {$opt(BOOT)} {
        boot_hw_device $dev
        set done 0
        for {set retry 0} {$retry < 30} {incr retry} {
            after 1000
            refresh_hw_device -update_hw_probes false $dev
            if {[get_property REGISTER.CONFIG_STATUS.BIT14_DONE_PIN $dev] eq "1"} {set done 1; break}
        }
        if {!$done} {error "Flash verified, but DONE did not assert after boot; check image/key and boot straps"}
        puts "BOOT VERIFIED: DONE asserted. Rescan PCIe separately if needed."
    } else {
        puts "Power-cycle the board to load the image, or rerun programming with --boot."
    }
}

set status [catch {run} message]
shutdown
if {$status} {
    puts stderr "ERROR: $message"
    puts stderr "If programming had started, inspect the reported stage before retrying; eFUSE cannot be undone."
    exit 1
}
set success [open SUCCESS w]
puts $success "ok"
close $success
exit 0
