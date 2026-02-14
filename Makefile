CC = gcc
CFLAGS = -Wall -O2 -g

all: user_dma_core

user_dma_core: user_dma_core.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f user_dma_core
	rm -f *.mcs *.prm *.jou *.log *.json *.csv
	rm -rf andromeda_IP-* andromeda_wrapper-*

load_drivers:
	sudo insmod /lib/modules/$$(uname -r)/xdma/xdma.ko interrupt_mode=0 config_bar_num=0
	sudo chmod 666 /dev/xdma*

run: user_dma_core
	sudo ./user_dma_core

rescan:
	sudo ./rescan_xilinx.sh

# Program FPGA with bit file
# Usage: make program FPGA_TARGET=<target> [BITFILE=<path_to_bit_file>]
# TARGET: alveo, puzhi, or alinx (default: alveo)
# BITFILE: optional custom bit file path (default: ../Vivado/FPGA_TARGET/FPGA_TARGET.runs/impl_1/andromeda_wrapper.bit)
TARGET ?= alveo
BITFILE ?= ../Vivado/$(TARGET)/$(TARGET).runs/impl_1/andromeda_wrapper.bit
program:
	@if [ ! -f "$(BITFILE)" ]; then \
		echo "Error: Bit file not found: $(BITFILE)"; \
		exit 1; \
	fi
	@bash -c 'set -o pipefail; vivado -mode batch -nolog -nojournal -source program_fpga.tcl -tclargs $(BITFILE) $(TARGET) 2>&1 | grep -v "^#"'
	sudo ./rescan_xilinx.sh

# Optional BITFILE= for program_flash: custom bitstream path (default from Vivado build)
program_flash:
	@bash -c 'set -o pipefail; vivado -mode batch -source program_flash.tcl -tclargs $(TARGET) $(if $(BITFILE),BITFILE=$(BITFILE),) 2>&1 | grep -v "^#"'

# Download artifact from GH Actions, extract bitstream from XSA, program FPGA
# Usage: make program_with_artifact GITHUB_RUN_ID=123456789 [TARGET=alveo]
GITHUB_RUN_ID ?= 18294195588
program_with_artifact:
	@# ── Check for existing extracted bitstream ──
	@BIT_FILE=$$(find . -maxdepth 4 -name "*.bit" \
	    -path "*/andromeda_wrapper*/xsa_extracted/*" 2>/dev/null | head -1); \
	if [ -n "$$BIT_FILE" ]; then \
		echo "✅ Found existing bitstream: $$BIT_FILE (skipping download)"; \
	else \
		echo "📥 Downloading artifacts from GitHub Actions run $(GITHUB_RUN_ID)..."; \
		gh run download $(GITHUB_RUN_ID); \
		echo "🔍 Locating XSA file..."; \
		XSA_FILE=$$(find . -maxdepth 3 -name "andromeda_wrapper.xsa" \
		    -path "*/andromeda_wrapper*" 2>/dev/null | head -1); \
		if [ -z "$$XSA_FILE" ]; then \
			echo "❌ No andromeda_wrapper.xsa found in downloaded artifacts"; \
			exit 1; \
		fi; \
		echo "📦 Found XSA: $$XSA_FILE"; \
		EXTRACT_DIR=$$(dirname "$$XSA_FILE")/xsa_extracted; \
		mkdir -p "$$EXTRACT_DIR"; \
		unzip -o "$$XSA_FILE" -d "$$EXTRACT_DIR"; \
		BIT_FILE=$$(find "$$EXTRACT_DIR" -name "*.bit" | head -1); \
		if [ -z "$$BIT_FILE" ]; then \
			echo "❌ No .bit file found inside XSA"; \
			exit 1; \
		fi; \
		echo "✅ Extracted bitstream: $$BIT_FILE"; \
	fi; \
	echo "🔧 Programming FPGA with $$BIT_FILE ..."; \
	$(MAKE) program TARGET=$(TARGET) BITFILE="$$BIT_FILE"

.PHONY: all clean load_drivers run rescan program program_flash program_with_artifact

