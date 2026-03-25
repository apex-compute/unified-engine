/*
 * user_dma_core.c — Minimal C example for Andromeda Unified Engine init & register access.
 *
 * Demonstrates:
 *   1. AXI-Lite register read/write via /dev/xdma0_user (pread/pwrite)
 *   2. DMA read/write via /dev/xdma0_h2c_0 and /dev/xdma0_c2h_0
 *   3. Engine initialization (latency delay registers, queue size)
 *   4. Reading the FPGA version register and dumping all engine registers
 *
 * Build:
 *   gcc -O2 -o user_dma_core user_dma_core.c
 *
 * Run:
 *   sudo ./user_dma_core
 */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>

/* ── Device paths ───────────────────────────────────────────────── */
#define DMA_DEVICE_H2C   "/dev/xdma0_h2c_0"
#define DMA_DEVICE_C2H   "/dev/xdma0_c2h_0"
#define DMA_DEVICE_USER  "/dev/xdma0_user"

/* ── Base address for AXI-Lite register space (matches Python UE_0_BASE_ADDR) */
#define UE_BASE_ADDR                 0x02000000
#define AXI_LITE_TRANSLATION_OFFSET  0x00000000

/* ── Register address offsets (from user_dma_core.py) ───────────── */
#define UE_FPGA_VERSION_ADDR         0x00000000
#define UE_START_ADDR                0x00000000
#define UE_ISA_CTRL                  0x00000004
#define UE_DRAM_ADDR                 0x00000008
#define UE_DMA_LENGTH_ADDR           0x0000000C
#define UE_CONTROL_ADDR              0x00000010
#define UE_STATUS_ADDR               0x00000014
#define UE_OUTPUT_VALID_DELAY_ADDR   0x0000001C
#define UE_URAM_LENGTH_ADDR          0x00000020
#define UE_URAM_WRITEBACK_ADDR       0x00000024
#define UE_LATENCY_COUNT_ADDR        0x00000030
#define UE_DRAM_URAM_CTRL_ADDR       0x00000034
#define UE_URAM_ROW_SIZE_ADDR        0x00000040
#define UE_VALID_DELAY_EXTRA_ADDR    0x00000044
#define UE_LALU_INST_ADDR            0x00000048
#define UE_LALU_DELAY_ADDR           0x0000004C
#define UE_SCALAR_ADDR               0x00000050
#define UE_QUEUE_CTRL_ADDR           0x00000054
#define UE_QUEUE_SIZE_ADDR           0x00000058
#define UE_URAM_LENGTH_ADDR_Z        0x0000005C
#define UE_BIAS_ADDER_EN_ADDR        0x00000060
#define UE_URAM_WB_PADDING_ADDR      0x00000064
#define UE_BROADCAST_MODE_ADDR       0x00000068
// #define UE_ARGMAX_INDEX              0x0000006C
#define UE_INSTRUCTION_CTL_ADDR      0x00000070
#define UE_TOTAL_BYTES_PER_STRIDE    0x00000074
#define UE_INSTRUCTION_ADDR          0x00000078
#define UE_TOTAL_STRIDE_BYTES        0x0000007C
#define UE_FMAX_CONTEXT_ADDR         0x00000080
#define UE_TRACE_BRAM_ADDR           0x00000084
#define UE_TRACE_BRAM_DATA           0x00000088
#define UE_ARGMAX1_INDEX             0x0000008C
#define UE_ARGMAX2_INDEX             0x00000090
#define UE_ARGMAX3_INDEX             0x00000094
#define UE_ARGMAX4_INDEX             0x00000098
#define UE_LAST_REG_ADDR             0x00000098

/* ── Latency constants (from user_dma_core.py pipeline calculations) ── */
#define UE_PIPELINE_BF19_MULT       2
#define UE_PIPELINE_BF19_ADD        3
#define UE_LATENCY_DOT_PRODUCT      21
#define UE_LATENCY_RMS              20
#define UE_LATENCY_EXP              24
#define UE_LATENCY_ADD_REDUCE       20
#define UE_LATENCY_ELTWISE_MUL      4
#define UE_LATENCY_ELTWISE_ADD      4
#define UE_LATENCY_ADD_EXP          8
#define UE_LATENCY_BF20_ITR2        3
#define UE_LATENCY_BF20_ITR3        11
#define UE_LATENCY_BF20_ITRGT3      11
#define UE_LALU_LATENCY_SOFTMAX     4
#define UE_LALU_LATENCY_RMS         7
#define UE_LALU_LATENCY_GELU        11
#define UE_LALU_LATENCY_SILU        10
#define UE_LATENCY_QSCALE           12
#define UE_QINPUT_DELAY             13
#define UE_LATENCY_QUANTIZATION     19

/* ── Memory constants ───────────────────────────────────────────── */
#define DRAM_START_ADDR              0x80000000

/* ── DMA read (Card → Host) ─────────────────────────────────────── */
static int dma_read(const char *device, uint64_t address, void *buffer, size_t size) {
    int fd = open(device, O_RDWR);
    if (fd < 0) { perror("dma_read: open"); return -1; }

    if (lseek(fd, address, SEEK_SET) != (off_t)address) {
        perror("dma_read: lseek"); close(fd); return -1;
    }

    ssize_t n = read(fd, buffer, size);
    close(fd);
    if (n < 0) { perror("dma_read: read"); return -1; }
    return (int)n;
}

/* ── DMA write (Host → Card) ────────────────────────────────────── */
static int dma_write(const char *device, uint64_t address, const void *buffer, size_t size) {
    int fd = open(device, O_RDWR);
    if (fd < 0) { perror("dma_write: open"); return -1; }

    if (lseek(fd, address, SEEK_SET) != (off_t)address) {
        perror("dma_write: lseek"); close(fd); return -1;
    }

    ssize_t n = write(fd, buffer, size);
    close(fd);
    if (n < 0) { perror("dma_write: write"); return -1; }
    return (int)n;
}

/* ── AXI-Lite register access via /dev/xdma0_user ───────────────── */
static void user_write_reg32(uint32_t reg_offset, uint32_t value) {
    int fd = open(DMA_DEVICE_USER, O_RDWR);
    if (fd < 0) { perror("user_write_reg32: open"); return; }

    off_t offset = reg_offset + UE_BASE_ADDR - AXI_LITE_TRANSLATION_OFFSET;
    if (pwrite(fd, &value, sizeof(value), offset) != sizeof(value))
        perror("user_write_reg32: pwrite");

    close(fd);
}

static uint32_t user_read_reg32(uint32_t reg_offset) {
    uint32_t value = 0xDEADBEEF;
    int fd = open(DMA_DEVICE_USER, O_RDWR);
    if (fd < 0) { perror("user_read_reg32: open"); return value; }

    off_t offset = reg_offset + UE_BASE_ADDR - AXI_LITE_TRANSLATION_OFFSET;
    if (pread(fd, &value, sizeof(value), offset) != sizeof(value))
        perror("user_read_reg32: pread");

    close(fd);
    return value;
}

/* ── Engine initialization (mirrors Python init_unified_engine) ── */
static void init_unified_engine(void) {
    printf("=== Unified Engine Initialization ===\n");

    /* 1. Read FPGA version register */
    uint32_t hw_version = user_read_reg32(UE_FPGA_VERSION_ADDR);
    printf("HW version: 0x%08X\n", hw_version);

    /* 2. Dump all engine registers */
    printf("\nEngine register dump:\n");
    for (uint32_t addr = UE_START_ADDR; addr <= UE_LAST_REG_ADDR; addr += 4) {
        uint32_t val = user_read_reg32(addr);
        if (val != 0xDEADBEEF)
            printf("  [0x%04X] = 0x%08X\n", addr, val);
    }

    /* 3. DRAM read/write self-test */
    printf("\nDRAM read/write test... ");
    uint32_t test_pattern = 0xCAFEBABE;
    dma_write(DMA_DEVICE_H2C, DRAM_START_ADDR, &test_pattern, sizeof(test_pattern));
    uint32_t readback = 0;
    dma_read(DMA_DEVICE_C2H, DRAM_START_ADDR, &readback, sizeof(readback));
    if (readback == test_pattern)
        printf("PASS (wrote 0x%08X, read 0x%08X)\n", test_pattern, readback);
    else
        printf("FAIL (wrote 0x%08X, read 0x%08X)\n", test_pattern, readback);

    /* 4. Configure output valid delay register
     *    Packs per-mode pipeline latencies into a single 32-bit word.
     *    See user_dma_core.py init_unified_engine() for field layout. */
    uint32_t ue_output_valid_delay =
        (UE_LATENCY_ADD_EXP      << 27) |
        (UE_LATENCY_RMS          << 22) |
        (UE_LATENCY_DOT_PRODUCT  << 17) |
        (UE_LATENCY_ELTWISE_ADD  << 13) |
        (UE_LATENCY_ELTWISE_MUL  <<  5) |
        (UE_LATENCY_EXP          <<  0);
    printf("\nWriting output_valid_delay = 0x%08X\n", ue_output_valid_delay);
    user_write_reg32(UE_OUTPUT_VALID_DELAY_ADDR, ue_output_valid_delay);

    /* 5. Configure extra delay register (bf20 adder / pipeline depths) */
    uint32_t ue_valid_delay_extra =
        (UE_LATENCY_BF20_ITRGT3      << 23) |
        (UE_LATENCY_BF20_ITR3        << 19) |
        (UE_LATENCY_BF20_ITR2        << 15) |
        ((UE_PIPELINE_BF19_MULT - 1) << 10) |
        ((UE_PIPELINE_BF19_ADD  - 1) <<  5) |
        (UE_LATENCY_ADD_REDUCE       <<  0);
    printf("Writing valid_delay_extra  = 0x%08X\n", ue_valid_delay_extra);
    user_write_reg32(UE_VALID_DELAY_EXTRA_ADDR, ue_valid_delay_extra);

    /* 6. Configure LALU (Last ALU) delay register */
    uint32_t ue_lalu_delay =
        (UE_QINPUT_DELAY          << 25) |
        (UE_LATENCY_QUANTIZATION  << 20) |
        (UE_LATENCY_QSCALE       << 16) |
        (UE_LALU_LATENCY_SILU    << 12) |
        (UE_LALU_LATENCY_GELU    <<  8) |
        (UE_LALU_LATENCY_RMS     <<  4) |
        (UE_LALU_LATENCY_SOFTMAX <<  0);
    printf("Writing lalu_delay         = 0x%08X\n", ue_lalu_delay);
    user_write_reg32(UE_LALU_DELAY_ADDR, ue_lalu_delay);

    /* 7. Set queue size to 1 (single-instruction queue) */
    user_write_reg32(UE_QUEUE_SIZE_ADDR, 1);

    printf("\nUnified Engine initialization complete.\n");
}

/* ── Main ────────────────────────────────────────────────────────── */
int main(void) {
    init_unified_engine();
    return 0;
}
