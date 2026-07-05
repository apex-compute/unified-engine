CC = gcc
CFLAGS = -Wall -O2 -g

all: user_dma_core

user_dma_core: user_dma_core.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f user_dma_core
	rm -f *.mcs *.prm *.jou *.log *.json *.csv
	rm -rf andromeda_IP-* andromeda_wrapper-*
	rm -f models/gemma3/gemma3_bin/gemma3_instruction*
	rm -f models/gemma3/gemma3_bin/gemma3_profile_instruction*
	rm -rf models/gpt2/gpt2_bin/
	rm -f models/llama3.2_1b/llama3.2_1b_bin/llama_instruction*
	rm -f models/llama3.2_1b/llama3.2_1b_bin/llama_profile_instruction*
	rm -f models/llama3.2_3b/llama3.2_3b_bin/llama3.2_3b_instruction_fpgapenalty.bin
	rm -f models/locateanything_3b/locateanything_3b_bin/programs*.bin
	rm -f models/locateanything_3b/locateanything_3b_bin/programs*.json
	rm -rf models/mobilesam/mobilesam_bin/
	rm -f models/parakeet/parakeet_bin/programs.bin
	rm -f models/parakeet/parakeet_bin/programs.json
	rm -f models/qwen2.5_vl_3b/qwen2.5_vl_3b_bin/programs.bin
	rm -f models/qwen2.5_vl_3b/qwen2.5_vl_3b_bin/programs.json
	rm -f models/qwen3_1.7b/qwen3_1.7b_bin/qwen3_1.7b_instruction*
	rm -f models/qwen3_1.7b/qwen3_1.7b_bin/programs.bin
	rm -f models/qwen3_1.7b/qwen3_1.7b_bin/programs.json
	rm -f models/qwen3_4b/qwen3_4b_bin/qwen3_4b_instruction*
	rm -f models/qwen3_4b/qwen3_4b_bin/programs.bin
	rm -f models/qwen3_4b/qwen3_4b_bin/programs.json
	rm -f models/qwen3.5_2b/qwen3.5_2b_bin/programs.bin
	rm -f models/qwen3.5_2b/qwen3.5_2b_bin/programs.json
	rm -f models/smolvlm2/smolvlm2_bin/programs*.bin
	rm -f models/smolvlm2/smolvlm2_bin/programs*.json
	rm -f models/smolvlm2/smolvlm2_bin/decoder_program.bin
	rm -f models/smolvlm2/smolvlm2_bin/decoder_program.json
	rm -rf models/swin/swin_bin/

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

# model_test run modes — pass extra words after the target:
#   make model_test                concise: banner + PASS/FAIL per model + final summary
#   make model_test verbose        also stream each model's live run log as it prints
#   make model_test nohup          run in background; output -> model_test_bg.out,
#                                  results -> model_auto_test_results.txt
#   make model_test gemma3         run only that model (any registered model name)
#   make model_test gemma3 llama3.2_1b verbose   combine: several models + a mode
# (make can't take a literal --verbose flag, so the modifiers are bare words.)
_MT_MODELS := $(shell python model_auto_test.py --list-names 2>/dev/null)
_MT_ARGS :=
ifneq (,$(filter verbose,$(MAKECMDGOALS)))
  _MT_ARGS := --verbose
endif
_MT_ONLY := $(filter $(_MT_MODELS),$(MAKECMDGOALS))
ifneq (,$(_MT_ONLY))
  _MT_ARGS += --only $(_MT_ONLY)
endif
# model_test cleans first by default; `run_from_bin` skips clean to reuse cached bins.
_MT_PREREQ := clean
ifneq (,$(filter run_from_bin,$(MAKECMDGOALS)))
  _MT_PREREQ :=
endif

model_test: $(_MT_PREREQ)
ifneq (,$(filter nohup,$(MAKECMDGOALS)))
	@nohup python model_auto_test.py $(_MT_ARGS) > model_test_bg.out 2>&1 & \
	echo "model_test started in background (PID $$!). Log -> model_test_bg.out, results -> model_auto_test_results.txt"
else
	python model_auto_test.py $(_MT_ARGS)
endif

# No-op modifier goals so `make model_test verbose|nohup|run_from_bin|<model-name>` doesn't error.
verbose nohup run_from_bin $(_MT_MODELS):
	@:

model_test_help:
	@echo "Usage: make model_test [MODEL...] [verbose] [nohup] [run_from_bin]"
	@echo ""
	@echo "Modes (bare words after the target):"
	@echo "  (none)    concise  : banner + PASS/FAIL per model + final summary"
	@echo "  verbose            : also stream each model's live run log as it prints"
	@echo "  nohup              : run in background -> model_test_bg.out,"
	@echo "                       results -> model_auto_test_results.txt"
	@echo "  run_from_bin       : skip 'make clean'; reuse existing compiled bins"
	@echo ""
	@echo "Models (run one or more by name; default = all):"
	@echo "  $(_MT_MODELS)"
	@echo ""
	@echo "Examples:"
	@echo "  make model_test                              # all models, concise"
	@echo "  make model_test gemma3                       # just one model"
	@echo "  make model_test gemma3 llama3.2_1b verbose   # two models + live log"
	@echo "  make model_test qwen3_4b nohup               # one model, in background"
	@echo "  make model_test gemma3 run_from_bin          # reuse cached bin, no rebuild"

_MT_NEW_MODELS := mobilesam locateanything_3b parakeet mobilenetv2_224 mobilenetv2_ssd_640 swin
model_test_new:
	@for m in $(_MT_NEW_MODELS); do \
		echo "--- Randomizing DRAM before $$m ---"; \
		python randomize_dram.py; \
		echo "--- Running $$m ---"; \
		python model_auto_test.py --only $$m; \
	done

.PHONY: all clean load_drivers run rescan program program_flash program_with_artifact model_test model_test_new model_test_help verbose nohup run_from_bin $(_MT_MODELS) $(_MT_NEW_MODELS)
