import os
import struct
import sys
import time

# ==========================================
# CONFIGURATION (exactly as in your working code)
# ==========================================
DMA_DEVICE_USER = "/dev/xdma0_user"
AXI_LITE_TRANSLATION_OFFSET  = 0x00000000
FLASH_BASE_ADDR = 0x00000000

BLOCK_SIZE       = 256 * 1024          # 256 KB erase blocks (datasheet for MT28GU512AAA1EGC-0SIT)
MAX_BUFFER_WORDS = 512                 # Max buffer size for Buffered Program
CHUNK_BYTES      = MAX_BUFFER_WORDS * 2  # 1 KB chunks
READ_DUMP_SIZE   = 256

# Micron Commands (from datasheet)
CMD_READ_ID          = 0x0090
CMD_READ_ARRAY       = 0x00FF
CMD_READ_STATUS      = 0x0070
CMD_CLEAR_STATUS     = 0x0050
CMD_BUFFERED_SETUP   = 0x00E9
CMD_BUFFERED_CONFIRM = 0x00D0


# ==========================================
# 32-BIT HELPERS (your exact working style)
# ==========================================
def user_write32(address: int, value: int):
    try:
        fd = os.open(DMA_DEVICE_USER, os.O_RDWR)
        try:
            offset = address - AXI_LITE_TRANSLATION_OFFSET
            data_bytes = struct.pack('I', value & 0xFFFFFFFF)
            os.pwrite(fd, data_bytes, offset)
        finally:
            os.close(fd)
    except Exception as e:
        print(f"Write32 error @ 0x{address:X} = 0x{value:08X}: {e}")


def user_read32(address: int) -> int:
    try:
        fd = os.open(DMA_DEVICE_USER, os.O_RDONLY)
        try:
            offset = address - AXI_LITE_TRANSLATION_OFFSET
            data_bytes = os.pread(fd, 4, offset)
            if len(data_bytes) == 4:
                return struct.unpack('I', data_bytes)[0]
            return 0
        finally:
            os.close(fd)
    except Exception as e:
        print(f"Read32 error @ 0x{address:X}: {e}")
        return 0


def user_read_bytes(address: int, size: int) -> bytes:
    try:
        fd = os.open(DMA_DEVICE_USER, os.O_RDONLY)
        try:
            offset = address - AXI_LITE_TRANSLATION_OFFSET
            return os.pread(fd, size, offset)
        finally:
            os.close(fd)
    except Exception as e:
        print(f"Bulk read error: {e}")
        return b''


# ==========================================
# STATUS & INFO
# ==========================================
def check_device_id():
    print("=== Reading Device ID ===")
    user_write32(FLASH_BASE_ADDR, CMD_CLEAR_STATUS)
    user_write32(FLASH_BASE_ADDR, CMD_READ_ID)

    raw = user_read32(FLASH_BASE_ADDR)
    manuf = raw & 0xFFFF
    dev   = (raw >> 16) & 0xFFFF

    manuf_ok = manuf == 0x0089
    dev_ok   = dev == 0x887E

    print(f"Manufacturer: 0x{manuf:04X}  {'PASS (Micron)' if manuf_ok else 'FAIL (expected Micron 0x0089)'}")
    print(f"Device ID:    0x{dev:04X}  {'PASS (MT28GU512AAA1EGC-0SIT)' if dev_ok else 'FAIL (expected 0x887E)'}")

    user_write32(FLASH_BASE_ADDR, CMD_READ_ARRAY)
    print("Back to Read Array mode.\n")
    return manuf_ok and dev_ok


def check_lock_status(block_addr):
    user_write32(block_addr, CMD_CLEAR_STATUS)
    user_write32(block_addr, CMD_READ_ID)
    lock_raw = user_read32(block_addr + 0x04)
    lock_bits = lock_raw & 0xFFFF
    user_write32(block_addr, CMD_READ_ARRAY)
    print(f"Lock status @ 0x{block_addr:X} = 0x{lock_bits:04X} {'(Unlocked)' if (lock_bits & 1) == 0 else '(Locked)'}")


def check_rcr_status(block_addr):
    user_write32(block_addr, CMD_CLEAR_STATUS)
    user_write32(block_addr, CMD_READ_ID)
    rcr_raw = user_read32(block_addr + 0x08)
    rcr = (rcr_raw >> 16) & 0xFFFF
    user_write32(block_addr, CMD_READ_ARRAY)
    print(f"RCR @ 0x{block_addr:X} = 0x{rcr:04X}")


# ==========================================
# UNLOCK + ERASE
# ==========================================
def flash_unlock_block(block_addr):
    print(f"[UNLOCK] Block @ 0x{block_addr:X}")
    user_write32(block_addr, CMD_CLEAR_STATUS)
    user_write32(block_addr, 0x00D00060)   # Unlock + Confirm
    user_write32(block_addr, CMD_READ_ARRAY)

    user_write32(block_addr, CMD_READ_ID)
    lock_bits = user_read32(block_addr + 0x04) & 0xFFFF
    user_write32(block_addr, CMD_READ_ARRAY)
    print(f"  → {'Unlocked' if (lock_bits & 1) == 0 else 'Still locked'} (0x{lock_bits:04X})\n")


def flash_erase_block(block_addr):
    flash_unlock_block(block_addr)
    print(f"[ERASE] Block @ 0x{block_addr:X} ...")

    user_write32(block_addr, CMD_CLEAR_STATUS)
    user_write32(block_addr, 0x00D00020)   # Erase + Confirm
    user_write32(block_addr, CMD_READ_ARRAY)
    user_write32(block_addr, CMD_READ_STATUS)

    timeout = 8000000
    status = 0
    start_time = time.time()
    while timeout > 0:
        status = user_read32(block_addr) & 0xFFFF
        if status & 0x0080:  # SR7 = Ready
            break
        time.sleep(0.001)
        timeout -= 1

    duration = time.time() - start_time
    if timeout <= 0 or (status & 0x003A):
        print(f"  Erase FAILED! Status=0x{status:04X} ({duration:.1f}s)")
        user_write32(block_addr, 0x00500050)
        return False
    else:
        print(f"  Erase OK ({duration:.1f}s)")
        user_write32(block_addr, CMD_READ_ARRAY)
        return True


# ==========================================
# BUFFERED PROGRAM (1 KB chunk with real data)
# ==========================================
def buffered_program_chunk(start_addr, data_words, num_words):
    if num_words < 1 or num_words > MAX_BUFFER_WORDS:
        num_words = min(num_words, MAX_BUFFER_WORDS)

    print(f"  [PROG] 0x{start_addr:X} ({num_words} words / {num_words*2} bytes)")

    # 1. Clear + Read Status
    user_write32(start_addr, 0x00700050)
    status = user_read32(start_addr) & 0xFFFF

    # 2. Setup
    setup_cmd = CMD_BUFFERED_SETUP | ((num_words - 1) << 16)
    user_write32(start_addr, setup_cmd)
    status = user_read32(start_addr) & 0xFFFF

    # 3. Data (32-bit packed = 2×16-bit words)
    for i in range(0, num_words, 2):
        w1 = data_words[i]
        w2 = data_words[i + 1] if i + 1 < num_words else 0xFFFF
        data_pair = w1 | (w2 << 16)
        user_write32(start_addr + i * 2, data_pair)

    # 4. Confirm + Read Status
    confirm_cmd = CMD_BUFFERED_CONFIRM | (CMD_READ_STATUS << 16)
    user_write32(start_addr, confirm_cmd)

    # 5. Poll
    timeout = 3000000
    while timeout > 0:
        status = user_read32(start_addr) & 0xFFFF
        if status & 0x0080:
            break
        time.sleep(0.000005)
        timeout -= 1

    if timeout <= 0 or (status & 0x003A):
        print(f"  Program FAILED! Status=0x{status:04X}")
        user_write32(start_addr, 0x00500050)
        return False

    user_write32(start_addr, 0x00FF00FF)  # Read Array
    return True


# ==========================================
# FULL PROGRAMMING (your .bit file)
# ==========================================
def program_flash_from_bit(bitfile_path):
    try:
        with open(bitfile_path, "rb") as f:
            bin_data = f.read()
    except Exception as e:
        print(f"Cannot open {bitfile_path}: {e}")
        return

    filesize = len(bin_data)
    print(f"\n=== PROGRAMMING {filesize:,} bytes ({filesize/1024/1024:.1f} MB) from {bitfile_path} ===\n")

    # === ERASE PHASE ===
    print("Erasing required 256 KB blocks...")
    end_addr = FLASH_BASE_ADDR + filesize
    blk = FLASH_BASE_ADDR
    while blk < end_addr: ### Change: end_addr
        if not flash_erase_block(blk):
            print("Erase failed - aborting!")
            return
        blk += BLOCK_SIZE

    # === PROGRAM PHASE ===
    print("\nStarting Buffered Programming (1 KB chunks)...")
    start_time = time.time()
    for offset in range(0, filesize, CHUNK_BYTES): ### Change: 1 to filesize
        chunk_size = min(CHUNK_BYTES, filesize - offset)
        num_words = (chunk_size + 1) // 2

        data_words = []
        for i in range(num_words):
            idx = offset + i * 2
            low  = bin_data[idx] if idx < filesize else 0xFF
            high = bin_data[idx + 1] if idx + 1 < filesize else 0xFF
            data_words.append(low | (high << 8))   # little-endian 16-bit

        chunk_start = FLASH_BASE_ADDR + offset
        if not buffered_program_chunk(chunk_start, data_words, num_words):
            print("Programming aborted due to error.")
            return

        if offset % (1024 * 1024) == 0 and offset > 0:   # progress every 1 MB
            print(f"  {offset // 1024:,} KB done...")

    duration = time.time() - start_time
    print(f"\n=== PROGRAMMING COMPLETE in {duration:.1f} seconds ===\n")


def verify_flash_with_bit(bitfile_path):
    try:
        with open(bitfile_path, "rb") as f:
            bin_data = f.read()
    except Exception as e:
        print(f"Cannot open {bitfile_path}: {e}")
        return

    filesize = len(bin_data)
    print(f"\n=== VERIFYING {filesize:,} bytes ({filesize/1024/1024:.1f} MB) from {bitfile_path} ===\n")

    # Enter Read Array mode
    user_write32(FLASH_BASE_ADDR, CMD_READ_ARRAY)

    # Verification in chunks (e.g., 1 KB, but read in 4-byte increments if needed)
    CHUNK_SIZE = 1024  # Arbitrary chunk size for progress
    mismatch = False
    start_time = time.time()

    for offset in range(0, filesize, CHUNK_SIZE):
        chunk_size = min(CHUNK_SIZE, filesize - offset)
        flash_data = bytearray()

        # Read chunk in 4-byte increments (as per constraint)
        cur_offset = FLASH_BASE_ADDR + offset
        remaining = chunk_size
        while remaining > 0:
            read_size = min(4, remaining)
            chunk = user_read_bytes(cur_offset, read_size)
            if len(chunk) != read_size:
                print(f"Read error at 0x{cur_offset:X}: Expected {read_size} bytes, got {len(chunk)}")
                mismatch = True
                break
            flash_data.extend(chunk)
            cur_offset += read_size
            remaining -= read_size

        if mismatch:
            break

        file_chunk = bin_data[offset:offset + chunk_size]
        if flash_data != file_chunk:
            for i in range(chunk_size):
                if i < len(flash_data) and flash_data[i] != file_chunk[i]:
                    print(f"Mismatch at 0x{FLASH_BASE_ADDR + offset + i:X}: Flash=0x{flash_data[i]:02X}, File=0x{file_chunk[i]:02X}")
                    mismatch = True
                    break
            if mismatch:
                break

        if offset % (1024 * 1024) == 0 and offset > 0:
            print(f"  {offset // 1024:,} KB verified...")

    duration = time.time() - start_time
    if mismatch:
        print(f"\n=== VERIFICATION FAILED in {duration:.1f} seconds ===\n")
    else:
        print(f"\n=== VERIFICATION SUCCESSFUL in {duration:.1f} seconds ===\n")

# ==========================================
# READ DUMP (your latest robust version)
# ==========================================
def read_flash_array_dump():
    print(f"=== Reading first {READ_DUMP_SIZE} bytes in Read Array mode ===")

    # Enter Read Array mode
    user_write32(FLASH_BASE_ADDR, CMD_READ_ARRAY)

    try:
        fd = os.open(DMA_DEVICE_USER, os.O_RDONLY)
    except Exception as e:
        print(f"Open failed: {e}")
        return

    try:
        offset = FLASH_BASE_ADDR - AXI_LITE_TRANSLATION_OFFSET
        if offset < 0:
            print("Invalid address offset")
            return

        # ---- Robust full read loop ----
        buf = bytearray()
        remaining = READ_DUMP_SIZE
        cur_offset = offset

        while remaining > 0:
            chunk = os.pread(fd, remaining, cur_offset)
            if not chunk:
                break
            buf.extend(chunk)
            n = len(chunk)
            remaining -= n
            cur_offset += n

        data = bytes(buf)

        if len(data) != READ_DUMP_SIZE:
            print(f"Warning: Expected {READ_DUMP_SIZE} bytes, got {len(data)} bytes")

        if not data:
            print("Bulk read failed.")
            return

        # ---- Print as 16-bit words ----
        print("\n---- Flash Dump (16-bit words) ----")

        for i in range(0, len(data), 32):  # 16 words per line (32 bytes)
            line = []
            chunk = data[i:i+32]

            for j in range(0, len(chunk), 2):
                # word = int.from_bytes(chunk[j:j+2], byteorder='little')
                word = int.from_bytes(chunk[j:j+2])
                line.append(f"{word:04X}")

            print(f"{i//2:04X}: " + " ".join(line))

    finally:
        os.close(fd)

    print("\n=== Read Array dump complete ===")


# ==========================================
# MAIN
# ==========================================
DEVICE_HASH_ADDR = 0x02000000

def read_device_hash():
    """Read hardware hash from flash at 0x02000000."""
    print(f"=== Reading Device Hash @ 0x{DEVICE_HASH_ADDR:08X} ===")
    data = user_read32(DEVICE_HASH_ADDR)
    print(f"Hardware hash: {data:08x}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MT28GU512 BPI Flash Programmer")
    parser.add_argument("bitfile", nargs="?", help=".bit or .bin file to program")
    parser.add_argument("--device", "-d", type=int, default=0,
                        help="FPGA device index (default: 0, maps to /dev/xdma<N>_user)")
    parser.add_argument("--check", action="store_true",
                        help="check device ID, flash status, and read FPGA git hash from flash")
    args = parser.parse_args()

    DMA_DEVICE_USER = f"/dev/xdma{args.device}_user"
    print(f"Using FPGA device: {DMA_DEVICE_USER}\n")

    if args.check:
        if args.bitfile:
            parser.error("--check takes no bitfile argument")
        print("=== MT28GU512AAA1EGC-0SIT BPI Flash Check ===\n")
        check_device_id()
        check_rcr_status(FLASH_BASE_ADDR)
        check_lock_status(FLASH_BASE_ADDR)
        read_device_hash()
        print("Check complete.")
        sys.exit(0)

    if not args.bitfile:
        parser.error("bitfile required (or use --check)")
        sys.exit(0)

    print("=== MT28GU512AAA1EGC-0SIT BPI Flash Programmer (Buffered Mode) ===\n")

    check_device_id()
    check_rcr_status(FLASH_BASE_ADDR)
    check_lock_status(FLASH_BASE_ADDR)
    program_flash_from_bit(args.bitfile)
    verify_flash_with_bit(args.bitfile)
    read_flash_array_dump()

    print("All done. You can now reboot or reconfigure the FPGA.")