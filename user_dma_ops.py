"""
Unified Engine ops: core + BF16/high-level (user_dma_ops).

Re-exports user_dma_core and extends UnifiedEngine with ops from unary_op_exp onward:
unary_op_exp, rope_operation, rope_operation_hf, allocate_*_dram, layer_norm,
eltwise_op, eltwise_add, eltwise_mul, broadcast_op, rms_norm, quantize_weight_to_dram,
quantized_matvec, matmat_mul_quantized_weights, patching, bf16_matmat, batched_matmat_mul,
matmat_mul, etc.

Usage:
    from user_dma_ops import UnifiedEngine, UE_MODE
    ue = UnifiedEngine(device='cpu')

For hardware tests: python user_hw_test.py [--dev xdma0] [--cycle 3.0]
"""

from user_dma_core import *  # noqa: F401, F403
import math

class Bf16OpsMixin:
        def unary_op_exp(self, size: int,
                         input_dram_addr: int,
                         output_dram_addr: int,
                         program_dram_addr: Optional[int] = None,
                         params_dram_addr: Optional[int] = None) -> Callable[[torch.Tensor], torch.Tensor]:
            """
            Create a reusable exponential approximation operation with instruction capture.
    
            This method performs exp(x) ≈ 1 + x + x^2/2 + x^3/6 using element-wise operations.
            Constant vectors (ones, 1/2, 1/6) are stored in params DRAM during setup and reused.
            Instructions are captured once during setup and reused.
    
            URAM Memory Map (row_size = size/64):
            =====================================
            URAM_A:                              URAM_B:
            ┌─────────────────────────┐          ┌─────────────────────────┐
            │ [0*rs] x (input)        │          │ [0*rs] x (copy)         │
            ├─────────────────────────┤          ├─────────────────────────┤
            │ [1*rs] x + 1            │          │ [1*rs] ones (1.0)       │ ◄─Const
            ├─────────────────────────┤          ├─────────────────────────┤
            │ [2*rs] x²               │          │ [2*rs] 1/2              │ ◄─Const
            ├─────────────────────────┤          ├─────────────────────────┤
            │ [3*rs] x³               │          │ [3*rs] 1/6              │ ◄─Const
            ├─────────────────────────┤          ├─────────────────────────┤
            │ [4*rs] x²/2             │          │ [4*rs] x³/6             │
            ├─────────────────────────┤          ├─────────────────────────┤
            │ [5*rs] result ◄─Output  │          │ [5*rs] x²/2 + x³/6      │
            └─────────────────────────┘          └─────────────────────────┘
    
            Computation Steps:
              1. x + 1       : A[x] + B[ones]  → A[x+1]
              2. x²          : A[x] * B[x]     → A[x²]
              3. x³          : A[x²] * B[x]    → A[x³]
              4. x²/2        : A[x²] * B[1/2]  → A[x²/2]
              5. x³/6        : A[x³] * B[1/6]  → B[x³/6]
              6. x²/2 + x³/6 : A[x²/2] + B[x³/6] → B[partial]
              7. result      : A[x+1] + B[partial] → A[result]
    
            Args:
                size: Number of elements in input vector (must be multiple of 64)
                input_dram_addr: DRAM address for input vector
                output_dram_addr: DRAM address for output vector
                program_dram_addr: DRAM address for instruction stream. If None, auto-allocated.
                params_dram_addr: DRAM address for parameters (ones, 1/2, 1/6 vectors). If None, auto-allocated.
    
            Returns:
                handler: Callable that takes input tensor and returns exp approximation result in bf16 format
            """
            bytes_per_element = 2
    
            # Verify alignment: size must be a multiple of UE_VECTOR_SIZE (64)
            if size % UE_VECTOR_SIZE != 0:
                raise ValueError(f"Vector size {size} must be a multiple of {UE_VECTOR_SIZE}")
    
            vector_bytes = size * bytes_per_element
    
            # Auto-allocate params_dram_addr if not provided
            use_auto_mem_management_for_params = False
            if params_dram_addr is None:
                use_auto_mem_management_for_params = True
                params_dram_addr = self.get_params_dram_addr()
            else:
                print(f"Using provided params DRAM address: {params_dram_addr}")
    
            # Create constant vectors and store in params DRAM
            ones_vector = torch.ones(size, dtype=torch.bfloat16, device=self.device)
            one_over_2_vector = torch.ones(size, dtype=torch.bfloat16, device=self.device) * (1.0 / 2.0)
            one_over_6_vector = torch.ones(size, dtype=torch.bfloat16, device=self.device) * (1.0 / 6.0)
    
            # Write constant vectors to params DRAM (stored once during setup)
            ones_vector_addr = params_dram_addr
            one_over_2_vector_addr = params_dram_addr + vector_bytes
            one_over_6_vector_addr = params_dram_addr + vector_bytes * 2
    
            self.dma_write(DMA_DEVICE_H2C, ones_vector_addr, ones_vector, vector_bytes)
            self.dma_write(DMA_DEVICE_H2C, one_over_2_vector_addr, one_over_2_vector, vector_bytes)
            self.dma_write(DMA_DEVICE_H2C, one_over_6_vector_addr, one_over_6_vector, vector_bytes)
    
            if use_auto_mem_management_for_params:
                self.allocate_params_dram(vector_bytes * 3)  # ones + 1/2 + 1/6
    
            # Start instruction capture
            self.start_capture()
            inst_id = 0
    
            # URAM address layout
            # BANK A
            uram_a_x_addr = URAM_START_ADDR
            uram_a_x_plus_1_addr = URAM_START_ADDR + size // UE_VECTOR_SIZE
            uram_a_x_squared_addr = URAM_START_ADDR + size // UE_VECTOR_SIZE * 2
            uram_a_x_cubed_addr = URAM_START_ADDR + size // UE_VECTOR_SIZE * 3
            uram_a_x_square_times_half_addr = URAM_START_ADDR + size // UE_VECTOR_SIZE * 4
            uram_a_result_addr = URAM_START_ADDR + size // UE_VECTOR_SIZE * 5
            # BANK B
            uram_b_x_addr = URAM_START_ADDR
            uram_b_ones_addr = URAM_START_ADDR + size // UE_VECTOR_SIZE
            uram_b_1_over_2_addr = URAM_START_ADDR + size // UE_VECTOR_SIZE * 2
            uram_b_1_over_6_addr = URAM_START_ADDR + size // UE_VECTOR_SIZE * 3
            uram_b_x_cube_times_1_over_6_addr = URAM_START_ADDR + size // UE_VECTOR_SIZE * 4
            uram_b_result_partial_addr = URAM_START_ADDR + size // UE_VECTOR_SIZE * 5
    
            # Copy input vector from DRAM to URAM_A
            self.ue_memcpy_from_dram(input_dram_addr, vector_bytes, 0, uram_a_x_addr, URAM_SECTION.URAM_A.value, inst_id)
            inst_id += 1
            self.wait_queue()
    
            # Copy input vector to URAM_B (needed for x^2 and x^3)
            self.ue_memcpy_from_dram(input_dram_addr, vector_bytes, 0, uram_b_x_addr, URAM_SECTION.URAM_B.value, inst_id)
            inst_id += 1
            self.wait_queue()
    
            # Copy constant vectors from params DRAM to URAM_B
            self.ue_memcpy_from_dram(ones_vector_addr, vector_bytes, 0, uram_b_ones_addr, URAM_SECTION.URAM_B.value, inst_id)
            inst_id += 1
            self.wait_queue()
    
            self.ue_memcpy_from_dram(one_over_2_vector_addr, vector_bytes, 0, uram_b_1_over_2_addr, URAM_SECTION.URAM_B.value, inst_id)
            inst_id += 1
            self.wait_queue()
    
            self.ue_memcpy_from_dram(one_over_6_vector_addr, vector_bytes, 0, uram_b_1_over_6_addr, URAM_SECTION.URAM_B.value, inst_id)
            inst_id += 1
            self.wait_queue()
    
            # Step 1: x + 1
            self.start_queue(
                0,  # broadcast_mode
                0,  # max_clear_en
                1,  # stride_z
                LALU_MODE.BYPASS.value,  # lalu_mode
                0,  # scalar
                0,  # uram_bram (URAM)
                URAM_SECTION.URAM_A.value,  # uram_section
                0,  # uram_dst_addr
                0,  # dram_to_uram_cpy_start
                uram_a_x_plus_1_addr,  # uram_wb_addr
                URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                UE_MODE.ELTWISE_ADD,  # mode
                0,  # data_type (not used)
                uram_a_x_addr,  # uram_a_start_addr
                uram_b_ones_addr,  # uram_b_start_addr
                size // UE_VECTOR_SIZE,  # uram_length
                0,  # dma_start_addr (not used)
                0,  # dma_length (not used)
                0,  # output_size (not used)
                inst_id  # inst_id
            )
            inst_id += 1
            self.wait_queue()
    
            # Step 2: x^2 = x * x
            self.start_queue(
                0,  # broadcast_mode
                0,  # max_clear_en
                1,  # stride_z
                LALU_MODE.BYPASS.value,  # lalu_mode
                0,  # scalar
                0,  # uram_bram (URAM)
                URAM_SECTION.URAM_A.value,  # uram_section
                0,  # uram_dst_addr
                0,  # dram_to_uram_cpy_start
                uram_a_x_squared_addr,  # uram_wb_addr
                URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                UE_MODE.ELTWISE_MUL,  # mode
                0,  # data_type (not used)
                uram_a_x_addr,  # uram_a_start_addr
                uram_b_x_addr,  # uram_b_start_addr
                size // UE_VECTOR_SIZE,  # uram_length
                0,  # dma_start_addr (not used)
                0,  # dma_length (not used)
                0,  # output_size (not used)
                inst_id  # inst_id
            )
            inst_id += 1
            self.wait_queue()
    
            # Step 3: x^3 = x^2 * x
            self.start_queue(
                0,  # broadcast_mode
                0,  # max_clear_en
                1,  # stride_z
                LALU_MODE.BYPASS.value,  # lalu_mode
                0,  # scalar
                0,  # uram_bram (URAM)
                URAM_SECTION.URAM_A.value,  # uram_section
                0,  # uram_dst_addr
                0,  # dram_to_uram_cpy_start
                uram_a_x_cubed_addr,  # uram_wb_addr
                URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                UE_MODE.ELTWISE_MUL,  # mode
                0,  # data_type (not used)
                uram_a_x_squared_addr,  # uram_a_start_addr
                uram_b_x_addr,  # uram_b_start_addr
                size // UE_VECTOR_SIZE,  # uram_length
                0,  # dma_start_addr (not used)
                0,  # dma_length (not used)
                0,  # output_size (not used)
                inst_id  # inst_id
            )
            inst_id += 1
            self.wait_queue()
    
            # Step 4: x^2 / 2 = x^2 * (1/2)
            self.start_queue(
                0,  # broadcast_mode
                0,  # max_clear_en
                1,  # stride_z
                LALU_MODE.BYPASS.value,  # lalu_mode
                0,  # scalar
                0,  # uram_bram (URAM)
                URAM_SECTION.URAM_A.value,  # uram_section
                0,  # uram_dst_addr
                0,  # dram_to_uram_cpy_start
                uram_a_x_square_times_half_addr,  # uram_wb_addr
                URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                UE_MODE.ELTWISE_MUL,  # mode
                0,  # data_type (not used)
                uram_a_x_squared_addr,  # uram_a_start_addr
                uram_b_1_over_2_addr,  # uram_b_start_addr
                size // UE_VECTOR_SIZE,  # uram_length
                0,  # dma_start_addr (not used)
                0,  # dma_length (not used)
                0,  # output_size (not used)
                inst_id  # inst_id
            )
            inst_id += 1
            self.wait_queue()
    
            # Step 5: x^3 / 6 = x^3 * (1/6)
            self.start_queue(
                0,  # broadcast_mode
                0,  # max_clear_en
                1,  # stride_z
                LALU_MODE.BYPASS.value,  # lalu_mode
                0,  # scalar
                0,  # uram_bram (URAM)
                URAM_SECTION.URAM_B.value,  # uram_section
                0,  # uram_dst_addr
                0,  # dram_to_uram_cpy_start
                uram_b_x_cube_times_1_over_6_addr,  # uram_wb_addr
                URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                UE_MODE.ELTWISE_MUL,  # mode
                0,  # data_type (not used)
                uram_a_x_cubed_addr,  # uram_a_start_addr
                uram_b_1_over_6_addr,  # uram_b_start_addr
                size // UE_VECTOR_SIZE,  # uram_length
                0,  # dma_start_addr (not used)
                0,  # dma_length (not used)
                0,  # output_size (not used)
                inst_id  # inst_id
            )
            inst_id += 1
            self.wait_queue()
    
            # Step 6: x^2/2 + x^3/6
            self.start_queue(
                0,  # broadcast_mode
                0,  # max_clear_en
                1,  # stride_z
                LALU_MODE.BYPASS.value,  # lalu_mode
                0,  # scalar
                0,  # uram_bram (URAM)
                URAM_SECTION.URAM_B.value,  # uram_section
                0,  # uram_dst_addr
                0,  # dram_to_uram_cpy_start
                uram_b_result_partial_addr,  # uram_wb_addr
                URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                UE_MODE.ELTWISE_ADD,  # mode
                0,  # data_type (not used)
                uram_a_x_square_times_half_addr,  # uram_a_start_addr
                uram_b_x_cube_times_1_over_6_addr,  # uram_b_start_addr
                size // UE_VECTOR_SIZE,  # uram_length
                0,  # dma_start_addr (not used)
                0,  # dma_length (not used)
                0,  # output_size (not used)
                inst_id  # inst_id
            )
            inst_id += 1
            self.wait_queue()
    
            # Step 7: (x + 1) + (x^2/2 + x^3/6) = final result
            self.start_queue(
                0,  # broadcast_mode
                0,  # max_clear_en
                1,  # stride_z
                LALU_MODE.BYPASS.value,  # lalu_mode
                0,  # scalar
                0,  # uram_bram (URAM)
                URAM_SECTION.URAM_A.value,  # uram_section
                0,  # uram_dst_addr
                0,  # dram_to_uram_cpy_start
                uram_a_result_addr,  # uram_wb_addr
                URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                UE_MODE.ELTWISE_ADD,  # mode
                0,  # data_type (not used)
                uram_a_x_plus_1_addr,  # uram_a_start_addr
                uram_b_result_partial_addr,  # uram_b_start_addr
                size // UE_VECTOR_SIZE,  # uram_length
                0,  # dma_start_addr (not used)
                0,  # dma_length (not used)
                0,  # output_size (not used)
                inst_id  # inst_id
            )
            inst_id += 1
            self.wait_queue()
    
            # Copy result from URAM to DRAM
            self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, uram_a_result_addr, output_dram_addr, vector_bytes, inst_id)
            self.wait_queue()
    
            # Finish capture and write instruction stream to DRAM
            self.stop_capture()
            self.generate_instruction_halt()
    
            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
                self.allocate_program_dram(program_size_bytes)
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
                self.write_captured_instructions_to_dram(program_dram_addr)
    
            def handler(input_vector):
                """
                Run exponential approximation using the captured instruction stream.
    
                Supports DeviceTensor for DMA cache optimization.
    
                Args:
                    input_vector: Input vector - torch.Tensor or DeviceTensor
    
                Returns:
                    DeviceTensor with exponential result, lazy C2H fetch
                """
                # Extract tensor data and check if DMA can be skipped
                if isinstance(input_vector, DeviceTensor):
                    input_data = input_vector._data
                    input_shape = input_data.shape
                    skip_dma = not input_vector.needs_dma(input_dram_addr)
                else:
                    input_data = input_vector
                    input_shape = input_vector.shape
                    skip_dma = False
    
                # Validate input
                assert input_data.dtype == torch.bfloat16, "Input vector must be in bf16 format"
                if input_data.numel() != size:
                    raise ValueError(f"Input vector has {input_data.numel()} elements, expected {size}")
    
                # Write input to DRAM (skip if DeviceTensor already synced)
                if skip_dma:
                    print(f"[DMA cache hit] Skipping input DMA to {hex(input_dram_addr)}")
                else:
                    if isinstance(input_vector, DeviceTensor):
                        input_vector.sync(input_dram_addr)
                    else: # if not a DeviceTensor, write the data to the FPGA DRAM
                        self.dma_write(DMA_DEVICE_H2C, input_dram_addr, input_data, vector_bytes)
    
                # Start executing from DRAM and wait for completion
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
    
                self.report_timing_and_instruction_count()
                # Exp approximation: ~10 ops per element (polynomial approximation)
                total_flops = 10 * size
                print(f"[exp_operation] {self.report_latency_in_us():.3f} us, {self.report_flop_rate_gflops(total_flops):.3f} GFLOPS")
    
                # Return DeviceTensor if input was DeviceTensor, else return raw tensor
                output_tensor = DeviceTensor(input_shape, ue=self, dram_addr=output_dram_addr)
                if isinstance(input_vector, DeviceTensor):
                    return output_tensor
                else:
                    return output_tensor.data
    
            return handler
    
        def rope_operation(self, size: int,
                          x_vector_dram_addr: int,
                          output_dram_addr: int,
                          rope_params: torch.Tensor,
                          program_dram_addr: Optional[int] = None,
                          params_dram_addr: Optional[int] = None) -> Callable[[torch.Tensor], torch.Tensor]:
            """
            Create a reusable RoPE (Rotary Position Embedding) operation with instruction capture.
    
            This method sets up a RoPE operation that can be called multiple times with different
            input vectors. Rope parameters are stored during setup (like weights) and reused for
            all subsequent calls. Instructions are captured once during setup and reused.
    
            URAM Memory Map (row_size = size/64):
            =====================================
            URAM_A:                              URAM_B:
            ┌─────────────────────────┐          ┌─────────────────────────┐
            │ 0x000 (URAM_START)      │          │ 0x000 (URAM_START)      │
            │ x (input vector)        │          │ theta (rope params)     │ ◄─Const
            │ [size elements]         │          │ [sin/cos interleaved]   │
            └─────────────────────────┘          └─────────────────────────┘
            ...
            ┌─────────────────────────┐
            │ 0x800 (URAM_HALFWAY)    │
            │ output (rotated x)      │ ◄─Result
            │ [size elements]         │
            └─────────────────────────┘
    
            Computation:
              ROPE mode with scalar=0x9FC00 (1.0 in bf19):
                - Applies complex rotation using theta parameters
                - x_rotated = view_as_real(view_as_complex(x) * cis(theta))
    
            Args:
                size: Number of elements in input vectors (must be multiple of 64)
                x_vector_dram_addr: DRAM address for input vector x
                output_dram_addr: DRAM address for output vector
                rope_params: Rope parameters (theta) tensor in bf16 format, containing sin/cos values.
                            This is stored in params DRAM during setup.
                program_dram_addr: DRAM address for instruction stream. If None, auto-allocated.
                params_dram_addr: DRAM address for parameters (rope_params). If None, auto-allocated.
    
            Returns:
                handler: Callable that takes input tensor (x_vector) and returns RoPE result in bf16 format
            """
            bytes_per_element = 2
    
            # Verify alignment: size must be a multiple of UE_VECTOR_SIZE (64)
            if size % UE_VECTOR_SIZE != 0:
                raise ValueError(f"Vector size {size} must be a multiple of {UE_VECTOR_SIZE}")
    
            # Validate rope_params
            assert rope_params.dtype == torch.bfloat16, "rope_params must be in bf16 format"
            if rope_params.numel() != size:
                raise ValueError(f"rope_params has {rope_params.numel()} elements, expected {size}")
    
    
            vector_bytes = size * bytes_per_element
            row_size = size // UE_VECTOR_SIZE
    
            # Auto-allocate params_dram_addr if not provided
            use_auto_mem_management_for_params = False
            if params_dram_addr is None:
                use_auto_mem_management_for_params = True
                params_dram_addr = self.get_params_dram_addr()
            else:
                print(f"Using provided params DRAM address: {params_dram_addr}")
    
            # Write rope_params to params DRAM (like weights, stored once during setup)
            rope_params_dram_addr = params_dram_addr
            self.dma_write(DMA_DEVICE_H2C, rope_params_dram_addr, rope_params, vector_bytes)
    
            if use_auto_mem_management_for_params:
                self.allocate_params_dram(vector_bytes)
    
            # Start instruction capture
            self.start_capture()
            inst_id = 0
    
            uram_x_addr = URAM_START_ADDR
            uram_theta_addr = URAM_START_ADDR
            uram_output_addr = URAM_HALFWAY_ADDR
    
            # Copy x vector from DRAM to URAM_A
            self.ue_memcpy_from_dram(x_vector_dram_addr, vector_bytes, 0, uram_x_addr, URAM_SECTION.URAM_A.value, inst_id)
            inst_id += 1
            self.wait_queue()
    
            # Copy rope parameters from DRAM to URAM_B
            self.ue_memcpy_from_dram(rope_params_dram_addr, vector_bytes, 0, uram_theta_addr, URAM_SECTION.URAM_B.value, inst_id)
            inst_id += 1
            self.wait_queue()
    
            # Perform ROPE operation
            # Scaling factor 0x9FC00 is 1.0 in bf19 format
            self.start_queue(
                0,  # broadcast_mode
                0,  # clear_max_en
                1,  # stride_z
                LALU_MODE.BYPASS.value,  # lalu_mode
                0x9FC00,  # BF19 scalar (1.0)
                0,  # uram_bram (URAM = 0)
                URAM_SECTION.URAM_A.value,  # uram_section (write to URAM_A)
                0,  # uram_dst_addr
                0,  # dram_to_uram_cpy_start
                uram_output_addr,  # uram_wb_addr (write result here)
                URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                UE_MODE.ROPE,  # mode
                0,  # data_type (not used)
                uram_x_addr,  # uram_a_start_addr (x vector)
                uram_theta_addr,  # uram_b_start_addr (rope params)
                row_size,  # uram_length (vector length / 64)
                0,  # dma_start_addr
                0,  # dma_length
                0,  # output_size
                inst_id  # inst_id
            )
            inst_id += 1
            self.wait_queue()
    
            # Copy result from URAM to DRAM
            self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, uram_output_addr, output_dram_addr, vector_bytes, inst_id)
            self.wait_queue()
    
            # Finish capture and write instruction stream to DRAM
            self.stop_capture()
            self.generate_instruction_halt()
    
            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
                self.allocate_program_dram(program_size_bytes)
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
                self.write_captured_instructions_to_dram(program_dram_addr)
    
            def handler(x_vector):
                """
                Run RoPE operation using the captured instruction stream.
    
                Supports DeviceTensor for DMA cache optimization.
    
                Args:
                    x_vector: Input vector - torch.Tensor or DeviceTensor
    
                Returns:
                    DeviceTensor with RoPE result, lazy C2H fetch
                """
                # Extract tensor data and check if DMA can be skipped
                if isinstance(x_vector, DeviceTensor):
                    input_data = x_vector._data
                    input_shape = input_data.shape
                    skip_dma = not x_vector.needs_dma(x_vector_dram_addr)
                else:
                    input_data = x_vector
                    input_shape = x_vector.shape
                    skip_dma = False
    
                # Validate input
                assert input_data.dtype == torch.bfloat16, "Input x_vector must be in bf16 format"
                if input_data.numel() != size:
                    raise ValueError(f"Input x_vector has {input_data.numel()} elements, expected {size}")
    
                # Write input to DRAM (skip if DeviceTensor already synced)
                if skip_dma:
                    print(f"[DMA cache hit] Skipping input DMA to {hex(x_vector_dram_addr)}")
                else:
                    if isinstance(x_vector, DeviceTensor):
                        x_vector.sync(x_vector_dram_addr)
                    else:
                        self.dma_write(DMA_DEVICE_H2C, x_vector_dram_addr, input_data, vector_bytes)
                # Start executing from DRAM and wait for completion
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
    
                self.report_timing_and_instruction_count()
                # RoPE: 4 ops per element (x*cos + x_rot*sin for each element pair)
                total_flops = 4 * size
                print(f"[rope_operation] {self.report_latency_in_us():.3f} us, {self.report_flop_rate_gflops(total_flops):.3f} GFLOPS")
    
                # Return DeviceTensor if input was DeviceTensor, else return raw tensor
                output_tensor = DeviceTensor(input_shape, ue=self, dram_addr=output_dram_addr)
                if isinstance(x_vector, DeviceTensor):
                    return output_tensor
                else:
                    return output_tensor.data
    
            return handler
    
        def rope_operation_hf(self, size: int,
                              x_vector_dram_addr: int,
                              output_dram_addr: int,
                              cos: torch.Tensor,
                              sin: torch.Tensor,
                              negate_sin: bool = True,
                              program_dram_addr: Optional[int] = None,
                              params_dram_addr: Optional[int] = None) -> Callable[[torch.Tensor], torch.Tensor]:
            """
            Create a reusable Hugging Face style RoPE operation with instruction capture.
    
            This method implements RoPE using element-wise operations:
            1) a = x_hf * cos
            2) b = x_hf[second_half] * sin[first_half]
            3) c = x_hf[first_half] * sin[second_half]
            4) d = a + cat(b, c)
    
            cos and sin are stored in params DRAM during setup (like weights) and reused.
    
            URAM Memory Map (row_size = size/64, half_row = row_size/2):
            ============================================================
            URAM_A:                              URAM_B:
            ┌─────────────────────────┐          ┌─────────────────────────┐
            │ [0*rs] x_hf (input)     │          │ [0*rs] cos (full)       │ ◄─Const
            ├─────────────────────────┤          ├─────────────────────────┤
            │ [1*rs] a = x_hf * cos   │          │ [1*rs] sin (full)       │ ◄─Const
            ├─────────────────────────┤          ├─────────────────────────┤
            │ [2*rs] d = result ◄─Out │          │ [2*rs] b = x[hi]*sin[lo]│ half
            └─────────────────────────┘          ├─────────────────────────┤
                                                 │ [2*rs+hr] c = x[lo]*sin[hi]│ half
                                                 └─────────────────────────┘
                                                 Note: b,c adjacent → cat(b,c)
    
            Computation Steps:
              1. a = x_hf * cos           : A[x] * B[cos]       → A[a]
              2. b = x[hi] * sin[lo]      : A[x+half] * B[sin]  → B[b]
              3. c = x[lo] * sin[hi]      : A[x] * B[sin+half]  → B[c]
              4. d = a + cat(b,c)         : A[a] + B[b]         → A[d]
                 (cat(b,c) = B[b] since b,c are contiguous)
    
            Args:
                size: Number of elements in input vector (must be multiple of 64)
                x_vector_dram_addr: DRAM address for input vector x_hf
                output_dram_addr: DRAM address for output vector
                cos: Cosine values tensor in bf16 format (size elements). Stored in params DRAM.
                sin: Sine values tensor in bf16 format (size elements). Stored in params DRAM.
                program_dram_addr: DRAM address for instruction stream. If None, auto-allocated.
                params_dram_addr: DRAM address for parameters (cos, sin). If None, auto-allocated.
    
            Returns:
                handler: Callable that takes input tensor (x_hf) and returns RoPE result in bf16 format
            """
            bytes_per_element = 2
    
            # Verify alignment: size must be a multiple of UE_VECTOR_SIZE (64)
            if size % UE_VECTOR_SIZE != 0:
                raise ValueError(f"Vector size {size} must be a multiple of {UE_VECTOR_SIZE}")
            if size % 2 != 0:
                raise ValueError(f"Vector size {size} must be even for HF RoPE")
    
            assert size >= 64, "Vector size must be at least 64 for HF RoPE"
            # Verify that cos and sin are the same size
            # Validate cos and sin
            assert cos.dtype == torch.bfloat16, "cos must be in bf16 format"
            assert sin.dtype == torch.bfloat16, "sin must be in bf16 format"
            if cos.numel() != size:
                raise ValueError(f"cos has {cos.numel()} elements, expected {size}")
            if sin.numel() != size:
                raise ValueError(f"sin has {sin.numel()} elements, expected {size}")
    
            vector_bytes = size * bytes_per_element
            half_size = size // 2
            row_size = size // UE_VECTOR_SIZE
            half_row_size = half_size // UE_VECTOR_SIZE
    
            # Auto-allocate params_dram_addr if not provided
            use_auto_mem_management_for_params = False
            if params_dram_addr is None:
                use_auto_mem_management_for_params = True
                params_dram_addr = self.get_params_dram_addr()
            else:
                print(f"Using provided params DRAM address: {params_dram_addr}")
    
            if negate_sin:
                sin[:half_size] = -sin[:half_size]
    
            cos_dram_addr = params_dram_addr
            sin_dram_addr = params_dram_addr + vector_bytes
            self.dma_write(DMA_DEVICE_H2C, cos_dram_addr, cos, vector_bytes)
            self.dma_write(DMA_DEVICE_H2C, sin_dram_addr, sin, vector_bytes)
    
            if use_auto_mem_management_for_params:
                self.allocate_params_dram(vector_bytes * 2)  # cos + sin
    
            # Start instruction capture
            self.start_capture()
            inst_id = 0
    
            # URAM address layout:
            # URAM_A: x_hf (full), a (full), d (full)
            # URAM_B: cos (full), sin (full), b (half), c (half), cat(b,c) (full)
    
            # x_hf is in the same URAM_A as a and d
            uram_x_addr = URAM_START_ADDR
            uram_a_addr = uram_x_addr + row_size
            uram_d_addr = uram_a_addr + row_size
    
            # cos and sin are in the same URAM_B
            uram_cos_addr = URAM_START_ADDR
            uram_sin_addr = uram_cos_addr + row_size
            uram_b_addr = uram_sin_addr + row_size
            uram_c_addr = uram_b_addr + half_row_size
    
            # Copy x_hf from DRAM to URAM_A
            self.ue_memcpy_from_dram(x_vector_dram_addr, vector_bytes, 0, uram_x_addr, URAM_SECTION.URAM_A.value, inst_id)
            inst_id += 1
            self.wait_queue()
    
            # Copy cos and sin from DRAM to URAM_B
            self.ue_memcpy_from_dram(cos_dram_addr, vector_bytes, 0, uram_cos_addr, URAM_SECTION.URAM_B.value, inst_id)
            inst_id += 1
            self.wait_queue()
    
            self.ue_memcpy_from_dram(sin_dram_addr, vector_bytes, 0, uram_sin_addr, URAM_SECTION.URAM_B.value, inst_id)
            inst_id += 1
            self.wait_queue()
    
            # Step 1: a = x_hf * cos
            self.start_queue(
                0,  # broadcast_mode
                0,  # max_clear_en
                1,  # stride_z
                LALU_MODE.BYPASS.value,  # lalu_mode
                0,  # scalar
                0,  # uram_bram (URAM)
                URAM_SECTION.URAM_A.value,  # uram_section (write to URAM_A)
                0,  # uram_dst_addr
                0,  # dram_to_uram_cpy_start
                uram_a_addr,  # uram_wb_addr (write a here)
                URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                UE_MODE.ELTWISE_MUL,  # mode
                0,  # data_type (not used)
                uram_x_addr,  # uram_a_start_addr (x_hf)
                uram_cos_addr,  # uram_b_start_addr (cos)
                row_size,  # uram_length
                0,  # dma_start_addr (not used)
                0,  # dma_length (not used)
                0,  # output_size (not used)
                inst_id  # inst_id
            )
            inst_id += 1
            self.wait_queue()
    
            # Step 2: b = x_hf[second_half] * sin[first_half]
            # x_hf[second_half] starts at uram_x_addr + half_row_size
            # sin[first_half] starts at uram_sin_addr
            self.start_queue(
                0,  # broadcast_mode
                0,  # max_clear_en
                1,  # stride_z
                LALU_MODE.BYPASS.value,  # lalu_mode
                0,  # scalar
                0,  # uram_bram (URAM)
                URAM_SECTION.URAM_B.value,  # uram_section (write to URAM_B)
                0,  # uram_dst_addr
                0,  # dram_to_uram_cpy_start
                uram_b_addr,  # uram_wb_addr (write b here)
                URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                UE_MODE.ELTWISE_MUL,  # mode
                0,  # data_type (not used)
                uram_x_addr + half_row_size,  # uram_a_start_addr (x_hf[second_half])
                uram_sin_addr,  # uram_b_start_addr (sin[first_half])
                half_row_size,  # uram_length
                0,  # dma_start_addr (not used)
                0,  # dma_length (not used)
                0,  # output_size (not used)
                inst_id  # inst_id
            )
            inst_id += 1
            self.wait_queue()
    
            # Step 3: c = x_hf[first_half] * sin[second_half]
            # x_hf[first_half] starts at uram_x_addr
            # sin[second_half] starts at uram_sin_addr + half_row_size
            self.start_queue(
                0,  # broadcast_mode
                0,  # max_clear_en
                1,  # stride_z
                LALU_MODE.BYPASS.value,  # lalu_mode
                0,  # scalar
                0,  # uram_bram (URAM)
                URAM_SECTION.URAM_B.value,  # uram_section (write to URAM_B)
                0,  # uram_dst_addr
                0,  # dram_to_uram_cpy_start
                uram_c_addr,  # uram_wb_addr (write c here)
                URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                UE_MODE.ELTWISE_MUL,  # mode
                0,  # data_type (not used)
                uram_x_addr,  # uram_a_start_addr (x_hf[first_half])
                uram_sin_addr + half_row_size,  # uram_b_start_addr (sin[second_half])
                half_row_size,  # uram_length
                0,  # dma_start_addr (not used)
                0,  # dma_length (not used)
                0,  # output_size (not used)
                inst_id  # inst_id
            )
            inst_id += 1
            self.wait_queue()
    
            # Step 4: Concatenate b and c -> cat(b, c) in URAM_B
            # b is at uram_b_addr, c is at uram_c_addr
            # We need to copy them to adjacent locations: uram_cat_bc_addr
            # First copy b to the first half of cat location
            # Note: We can't directly concatenate in hardware, so we'll write b and c to adjacent locations
            # Actually, b and c are already adjacent (b at uram_b_addr, c at uram_c_addr = uram_b_addr + half_row_size)
            # So cat(b, c) is already at uram_b_addr spanning half_row_size * 2 = row_size
            # critical step!!!
            uram_cat_bc_addr = uram_b_addr  # b and c are already concatenated in URAM
    
            # Step 5: d = a + cat(b, c)
            self.start_queue(
                0,  # broadcast_mode
                0,  # max_clear_en
                1,  # stride_z
                LALU_MODE.BYPASS.value,  # lalu_mode
                0,  # scalar
                0,  # uram_bram (URAM)
                URAM_SECTION.URAM_A.value,  # uram_section (write to URAM_A)
                0,  # uram_dst_addr
                0,  # dram_to_uram_cpy_start
                uram_d_addr,  # uram_wb_addr (write d here)
                URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                UE_MODE.ELTWISE_ADD,  # mode
                0,  # data_type (not used)
                uram_a_addr,  # uram_a_start_addr (a)
                uram_cat_bc_addr,  # uram_b_start_addr (cat(b, c))
                row_size,  # uram_length
                0,  # dma_start_addr (not used)
                0,  # dma_length (not used)
                0,  # output_size (not used)
                inst_id  # inst_id
            )
            inst_id += 1
            self.wait_queue()
    
            # Copy result from URAM to DRAM
            self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, uram_d_addr, output_dram_addr, vector_bytes, inst_id)
            self.wait_queue()
    
            # Finish capture and write instruction stream to DRAM
            self.stop_capture()
            self.generate_instruction_halt()
    
            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
                self.allocate_program_dram(program_size_bytes)
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
                self.write_captured_instructions_to_dram(program_dram_addr)
    
            def handler(x_hf):
                """
                Run Hugging Face style RoPE operation using the captured instruction stream.
    
                Supports DeviceTensor for DMA cache optimization.
    
                Args:
                    x_hf: Input vector - torch.Tensor or DeviceTensor
    
                Returns:
                    DeviceTensor with RoPE result, lazy C2H fetch
                """
                # Extract tensor data and check if DMA can be skipped
                if isinstance(x_hf, DeviceTensor):
                    input_data = x_hf._data
                    input_shape = input_data.shape
                    skip_dma = not x_hf.needs_dma(x_vector_dram_addr)
                else:
                    input_data = x_hf
                    input_shape = x_hf.shape
                    skip_dma = False
    
                # Validate input
                assert input_data.dtype == torch.bfloat16, "Input x_hf must be in bf16 format"
                if input_data.numel() != size:
                    raise ValueError(f"Input x_hf has {input_data.numel()} elements, expected {size}")
    
                # Write input to DRAM (skip if DeviceTensor already synced)
                if skip_dma:
                    print(f"[DMA cache hit] Skipping input DMA to {hex(x_vector_dram_addr)}")
                else:
                    if isinstance(x_hf, DeviceTensor):
                        x_hf.sync(x_vector_dram_addr)
                    else:
                        self.dma_write(DMA_DEVICE_H2C, x_vector_dram_addr, input_data, vector_bytes)
    
                # Start executing from DRAM and wait for completion
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
    
                self.report_timing_and_instruction_count()
                # RoPE HF: 4 ops per element (x*cos + x_rot*sin for each element pair)
                total_flops = 4 * size
                print(f"[rope_operation_hf] {self.report_latency_in_us():.3f} us, {self.report_flop_rate_gflops(total_flops):.3f} GFLOPS")
    
                # Return DeviceTensor if input was DeviceTensor, else return raw tensor
                output_tensor = DeviceTensor(input_shape, ue=self, dram_addr=output_dram_addr)
                if isinstance(x_hf, DeviceTensor):
                    return output_tensor
                else:
                    return output_tensor.data
    
            return handler
    
        def layer_norm(self, shape: Tuple[int, int],
                       input_dram_addr: int,
                       output_dram_addr: int,
                       program_dram_addr: Optional[int] = None,
                       params_dram_addr: Optional[int] = None,
                       gamma: torch.Tensor = None,
                       beta: torch.Tensor = None) -> Callable[[torch.Tensor], torch.Tensor]:
            """
            Perform Layer Normalization with scalar on device
    
            This method performs layer normalization:
            1) Calculate E[X] = sum(x)/n
            2) Calculate z = x - E[x]
            3) Do the RMS normalization for z: y = z/rms[z]
            4) Optionally multiply by gamma: y = y * gamma
            5) Optionally add beta: y = y + beta
    
            URAM Memory Map (row_size = norm_dim/64):
            ==========================================
            URAM_A:                                  URAM_B:
            ┌──────────────────────────────┐         ┌─────────────────────────┐
            │ 0x000                        │         │ 0x000                   │
            │ x[0] (1st vector) [row_size] │         │ zeros [row_size]        │ ◄─Const
            ├──────────────────────────────┤         ├─────────────────────────┤
            │ x[1] (2nd vector)            │         │ gamma (γ) [row_size]    │ ◄─Const
            ├──────────────────────────────┤         ├─────────────────────────┤
            │ ...                          │         │ beta (β) [row_size]     │ ◄─Const
            ├──────────────────────────────┤         └─────────────────────────┘
            │ (chunk limit ~0xF00)         │
            └──────────────────────────────┘
            ...
            ┌──────────────────────────────┐
            │ URAM_END - row_size          │
            │ x - E[x] (temp scratch)      │ ◄─uram_output_addr
            └──────────────────────────────┘
    
            Per-Vector Computation Steps:
              1. ADD_REDUCE     : sum(x[i])           → LALU (E[x] = sum/n)
              2. ADD_BROADCAST  : x[i] - E[x]         → A[output_addr]
              3. RMS            : rms(x-E[x])         → LALU (1/rms)
              4. MUL_BROADCAST  : (x-E[x]) * (1/rms)  → A[i] (normalized)
              5. ELTWISE_MUL    : norm * γ            → A[i] (if gamma)
              6. ELTWISE_ADD    : result + β          → A[i] (if beta)
    
            Args:
                shape: Tuple of (batch_size, norm_dim) specifying the input shape
                input_dram_addr: DRAM address for input data
                output_dram_addr: DRAM address for output data
                program_dram_addr: DRAM address for instruction stream. If None, auto-allocated.
                params_dram_addr: DRAM address for parameters (zeros, gamma, beta). If None, auto-allocated.
                gamma: Optional gamma (scale) parameter vector in bf16 format. If None, gamma is not applied.
                beta: Optional beta (shift) parameter vector in bf16 format. If None, beta is not applied.
    
            Returns:
                handler: Callable that takes input tensor and returns layer normalized result in bf16 format
            """
            norm_dim = shape[1]
            bytes_per_element = 2
    
            if (norm_dim * bytes_per_element) > URAM_HALF_WAY_SIZE:
                raise ValueError(f"Norm dimension {norm_dim} must be less than {URAM_HALF_WAY_SIZE}")
    
            size = shape[0] * shape[1]
    
            # Verify alignment: size must be a multiple of UE_VECTOR_SIZE (64)
            if norm_dim % UE_VECTOR_SIZE != 0:
                raise ValueError(f"Vector size {norm_dim} must be a multiple of {UE_VECTOR_SIZE}")
    
            # Validate gamma and beta if provided
            if gamma is not None:
                assert gamma.dtype == torch.bfloat16, "Gamma must be in bf16 format"
                assert gamma.numel() == norm_dim, f"Gamma size {gamma.numel()} must match input size {norm_dim}"
            if beta is not None:
                assert beta.dtype == torch.bfloat16, "Beta must be in bf16 format"
                assert beta.numel() == norm_dim, f"Beta size {beta.numel()} must match input size {norm_dim}"
    
            vector_bytes = norm_dim * bytes_per_element
            row_size = norm_dim // UE_VECTOR_SIZE  # Number of 64-element rows
            total_bytes = size * bytes_per_element
    
            # Auto-allocate params_dram_addr if not provided
            use_auto_mem_management_for_params = False
            if params_dram_addr is None:
                use_auto_mem_management_for_params = True
                params_dram_addr = self.get_params_dram_addr()
            else:
                print(f"Using provided params DRAM address: {params_dram_addr}")
    
            size_for_secondary_memory_transfers = 0
            # Fill zeros to DRAM (x + 0 = x for mean calculation)
            zeros_vector = torch.zeros(size, dtype=torch.bfloat16, device=self.device)
            self.dma_write(DMA_DEVICE_H2C, params_dram_addr, zeros_vector, vector_bytes)
            size_for_secondary_memory_transfers += vector_bytes
    
            gamma_dram_addr = None
            if gamma is not None:
                # Write gamma to DRAM
                gamma_dram_addr = params_dram_addr + vector_bytes
                self.dma_write(DMA_DEVICE_H2C, gamma_dram_addr, gamma, vector_bytes)
                size_for_secondary_memory_transfers += vector_bytes
    
            beta_dram_addr = None
            if beta is not None:
                # Write beta to DRAM
                if gamma is None:
                    beta_dram_addr = params_dram_addr + vector_bytes
                else:
                    beta_dram_addr = gamma_dram_addr + vector_bytes
    
                self.dma_write(DMA_DEVICE_H2C, beta_dram_addr, beta, vector_bytes)
                size_for_secondary_memory_transfers += vector_bytes
    
            if use_auto_mem_management_for_params:
                self.allocate_params_dram(size_for_secondary_memory_transfers)
    
            result_dram_addr = output_dram_addr
            in_dram_addr = input_dram_addr
            uram_a_start_addr = URAM_START_ADDR
            uram_b_start_addr = URAM_START_ADDR
            uram_output_addr = URAM_END_ADDR - row_size
    
            self.start_capture()
            inst_id = 0
            self.ue_memcpy_from_dram(params_dram_addr, size_for_secondary_memory_transfers, 0, uram_b_start_addr, URAM_SECTION.URAM_B.value, inst_id)
            inst_id += 1
            self.wait_queue()
    
            remaining_bytes = total_bytes
            while remaining_bytes > 0:
                chunk_bytes = min(remaining_bytes, URAM_0XF00_SIZE)
                # Align chunk_bytes to norm_dim * bytes_per_element
                chunk_bytes = (chunk_bytes // (norm_dim * bytes_per_element)) * norm_dim * bytes_per_element
    
                remaining_bytes -= chunk_bytes
                #   Copy input vector from DRAM to URAM_A
    
                self.ue_memcpy_from_dram(in_dram_addr, chunk_bytes, 0, uram_a_start_addr, URAM_SECTION.URAM_A.value, inst_id)
                inst_id += 1
                self.wait_queue()
    
                in_dram_addr += chunk_bytes
                number_of_norms = chunk_bytes // (norm_dim * bytes_per_element)
    
                for i in range(number_of_norms):
                    # Step 1: Calculate mean E[X] = sum(x)/n
                    # Element-wise add to calculate the mean, accumulator is enabled
                    # E[x] = sum(x)/n is cached in LALU (1/n is the scalar)
                    scalar_mean = self.float_to_bf19(float(norm_dim))
                    self.start_queue(
                        0,  # broadcast_mode
                        0,  # clear_max_en
                        1,  # stride_z
                        LALU_MODE.MODE_RECIP.value,  # lalu_mode (computes sum/n)
                        scalar_mean,  # BF19 scalar (n)
                        0,  # uram_bram (URAM = 0)
                        URAM_SECTION.URAM_A.value,  # uram_section
                        0,  # uram_dst_addr
                        0,  # dram_to_uram_cpy_start
                        0,  # uram_wb_addr (no writeback, result in LALU)
                        URAM_WRITE_SRC.URAM_WB_DISABLE.value,  # NO WRITEBACK
                        UE_MODE.ADD_REDUCE,  # mode
                        0,  # data_type not used
                        uram_a_start_addr,  # uram_a_start_addr
                        uram_b_start_addr,  # uram_b_start_addr, zeros vector
                        row_size,  # uram_length (row_size)
                        0,  # dma_start_addr
                        0,  # dma_length
                        0,  # output_size
                        inst_id   # inst_id
                    )
                    inst_id += 1
                    self.wait_queue()
    
                    # Step 2: Calculate x - E[x] using broadcast subtraction
                    self.start_queue(
                        BROADCAST_MODE.LALU_RESULT_NEGATE.value,  # broadcast_mode
                        0,  # clear_max_en
                        1,  # stride_z
                        LALU_MODE.BYPASS.value,  # lalu_mode
                        0,  # scalar (not used)
                        0,  # uram_bram (URAM = 0)
                        URAM_SECTION.URAM_A.value,  # uram_section
                        0,  # uram_dst_addr
                        0,  # dram_to_uram_cpy_start
                        uram_output_addr,  # uram_wb_addr (write x - E[x] to URAM_A)
                        URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                        UE_MODE.ADD_BROADCAST,  # mode (broadcast subtraction)
                        0,  # data_type not used
                        uram_a_start_addr,  # uram_a_start_addr
                        0,  # uram_b_start_addr (not used in broadcast subtraction)
                        row_size,  # uram_length
                        0,  # dma_start_addr
                        0,  # dma_length
                        0,  # output_size
                        inst_id   # inst_id
                    )
                    self.wait_queue()
                    inst_id += 1
    
                    # Step 3: Calculate RMS of (x - E[x])
                    # Test sum of squaring, 1/sqrt(sum((x - E[x])^2)) is cached
                    sqrt_n_vector = math.sqrt(norm_dim)
                    scalar_rms = self.float_to_bf19(sqrt_n_vector)
                    self.start_queue(
                        0,  # broadcast_mode
                        0,  # clear_max_en
                        1,  # stride_z
                        LALU_MODE.MODE_RSQRT.value,  # lalu_mode
                        scalar_rms,  # BF19 scalar (sqrt(UE_VECTOR_SIZE))
                        0,  # uram_bram (URAM = 0)
                        URAM_SECTION.URAM_A.value,  # uram_section
                        0,  # uram_dst_addr
                        0,  # dram_to_uram_cpy_start
                        0,  # uram_wb_addr (no writeback, result in LALU)
                        URAM_WRITE_SRC.URAM_WB_DISABLE.value,  # uram_write_src
                        UE_MODE.RMS,  # mode
                        0,  # data_type not used
                        uram_output_addr,  # uram_a_start_addr
                        0,  # uram_b_start_addr, not used in RMS
                        row_size,  # uram_length
                        0,  # dma_start_addr
                        0,  # dma_length
                        0,  # output_size
                        inst_id   # inst_id
                    )
                    self.wait_queue()
                    inst_id += 1
    
                    # Step 4: Broadcast multiply to normalize
                    # Multiply (x - E[x]) by 1/rms[(x - E[x])] using cached value from LALU
                    self.start_queue(
                        BROADCAST_MODE.LALU_RESULT.value,  # broadcast_mode
                        0,  # clear_max_en
                        1,  # stride_z
                        LALU_MODE.BYPASS.value,  # lalu_mode
                        0,  # scalar (not used)
                        0,  # uram_bram (URAM = 0)
                        URAM_SECTION.URAM_A.value,  # uram_section
                        0,  # uram_dst_addr
                        0,  # dram_to_uram_cpy_start
                        uram_a_start_addr,  # uram_wb_addr (write final result here)
                        URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                        UE_MODE.MUL_BROADCAST,  # mode
                        0,  # data_type not used
                        uram_output_addr,  # uram_a_start_addr
                        0,  # uram_b_start_addr not used in broadcast multiply
                        row_size,  # uram_length
                        0,  # dma_start_addr
                        0,  # dma_length
                        0,  # output_size
                        inst_id   # inst_id
                    )
                    self.wait_queue()
                    inst_id += 1
    
                    # Step 5 (optional): Multiply by gamma if provided
                    if gamma is not None:
                        # Element-wise multiply: result = result * gamma
                        self.start_queue(
                            0,  # broadcast_mode
                            0,  # clear_max_en
                            1,  # stride_z
                            LALU_MODE.BYPASS.value,  # lalu_mode
                            0,  # scalar (not used)
                            0,  # uram_bram (URAM = 0)
                            URAM_SECTION.URAM_A.value,  # uram_section
                            0,  # uram_dst_addr
                            0,  # dram_to_uram_cpy_start
                            uram_a_start_addr,  # uram_wb_addr (write result back)
                            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                            UE_MODE.ELTWISE_MUL,  # mode
                            0,  # data_type not used
                            uram_a_start_addr,  # uram_a_start_addr (normalized result)
                            uram_b_start_addr + (vector_bytes // (UE_VECTOR_SIZE * bytes_per_element)),  # uram_b_start_addr (gamma)
                            row_size,  # uram_length
                            0,  # dma_start_addr
                            0,  # dma_length
                            0,  # output_size
                            inst_id  # inst_id
                        )
                        self.wait_queue()
                        inst_id += 1
    
                    # Step 6 (optional): Add beta if provided
                    if beta is not None:
                        # Element-wise add: result = result + beta
                        self.start_queue(
                            0,  # broadcast_mode
                            0,  # clear_max_en
                            1,  # stride_z
                            LALU_MODE.BYPASS.value,  # lalu_mode
                            0,  # scalar (not used)
                            0,  # uram_bram (URAM = 0)
                            URAM_SECTION.URAM_A.value,  # uram_section
                            0,  # uram_dst_addr
                            0,  # dram_to_uram_cpy_start
                            uram_a_start_addr,  # uram_wb_addr (write result back)
                            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                            UE_MODE.ELTWISE_ADD,  # mode
                            0,  # data_type not used
                            uram_a_start_addr,  # uram_a_start_addr (result so far)
                            uram_b_start_addr + (vector_bytes // (UE_VECTOR_SIZE * bytes_per_element)) * 2,  # uram_b_start_addr (beta)
                            row_size,  # uram_length
                            0,  # dma_start_addr
                            0,  # dma_length
                            0,  # output_size
                            inst_id  # inst_id
                        )
                        self.wait_queue()
                        inst_id += 1
    
                    uram_a_start_addr += row_size
    
                uram_a_start_addr = URAM_START_ADDR
                self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, uram_a_start_addr, result_dram_addr, chunk_bytes, inst_id)
                self.wait_queue()
                result_dram_addr += chunk_bytes
    
            # Finish capture and write instruction stream to DRAM once
            self.stop_capture()
            self.generate_instruction_halt()
    
            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
                self.allocate_program_dram(program_size_bytes)
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
                self.write_captured_instructions_to_dram(program_dram_addr)
    
            def handler(input_tensor):
                """
                Run layer normalization using the captured instruction stream.
    
                Supports DeviceTensor for DMA cache optimization.
    
                Args:
                    input_tensor: Input tensor - torch.Tensor or DeviceTensor
                """
                # Extract tensor data and check if DMA can be skipped
                if isinstance(input_tensor, DeviceTensor):
                    input_data = input_tensor._data
                    input_shape = input_data.shape
                    skip_dma = not input_tensor.needs_dma(input_dram_addr)
                else:
                    input_data = input_tensor
                    input_shape = input_tensor.shape
                    skip_dma = False
    
                # Validate input
                assert input_data.dtype == torch.bfloat16, "Input tensor must be in bf16 format"
                if input_data.numel() != size:
                    raise ValueError(f"Input tensor has {input_data.numel()} elements, expected {size}")
    
                # Write input to DRAM (skip if DeviceTensor already synced)
                if skip_dma:
                    print(f"[DMA cache hit] Skipping input DMA to {hex(input_dram_addr)}")
                else:
                    if isinstance(input_tensor, DeviceTensor):
                        input_tensor.sync(input_dram_addr)
                    else:
                        self.dma_write(DMA_DEVICE_H2C, input_dram_addr, input_data, total_bytes)
    
                # Start executing from DRAM and wait for completion
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
    
                self.report_timing_and_instruction_count()
                # LayerNorm FLOPs: mean(N), subtract(N), variance(2N), rsqrt(1), scale(N), gamma(N), beta(N)
                # Approximately 5*N per vector + optional gamma(N) + optional beta(N)
                batch_size = shape[0]
                flops_per_vector = 5 * norm_dim
                if gamma is not None:
                    flops_per_vector += norm_dim
                if beta is not None:
                    flops_per_vector += norm_dim
                total_flops = batch_size * flops_per_vector
                print(f"[layer_norm] {self.report_latency_in_us():.3f} us, {self.report_flop_rate_gflops(total_flops):.3f} GFLOPS")
    
                # Return DeviceTensor if input was DeviceTensor, else return raw tensor
                output_tensor = DeviceTensor(input_shape, ue=self, dram_addr=output_dram_addr)
                if isinstance(input_tensor, DeviceTensor):
                    return output_tensor
                else:
                    return output_tensor.data
    
            return handler
    
        def eltwise_op(self, size: int,
                       eltwise_type: UE_MODE,
                       input_a_dram_addr: int,
                       input_b_dram_addr: int,
                       output_dram_addr: int,
                       program_dram_addr: Optional[int] = None,
                       params_dram_addr: Optional[int] = None) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
            """
            Create a reusable element-wise operation with instruction capture.
    
            This method sets up an element-wise operation (add or multiply) that can be
            called multiple times with different inputs. Instructions are captured once
            during setup and reused for subsequent calls.
    
            URAM Memory Map:
            ================
            URAM_A:                          URAM_B:
            ┌─────────────────────┐          ┌─────────────────────┐
            │ 0x000               │          │ 0x000               │
            │ input_a [row_size]  │ ◄─Read   │ input_b [row_size]  │ ◄─Read
            │                     │          │                     │
            ├─────────────────────┤          └─────────────────────┘
            │ result [row_size]   │ ◄─Write
            │ (overwrites input_a)│
            └─────────────────────┘
    
            Data Flow:
              DRAM → URAM_A[0x000] (input_a)
              DRAM → URAM_B[0x000] (input_b)
              Execute: URAM_A[0x000] ⊕ URAM_B[0x000] → URAM_A[0x000]
              URAM_A[0x000] → DRAM (result)
    
            Args:
                size: Number of elements in input vectors (must be multiple of 64)
                eltwise_type: Operation type (UE_MODE.ELTWISE_ADD or UE_MODE.ELTWISE_MUL)
                input_a_dram_addr: DRAM address for first input vector
                input_b_dram_addr: DRAM address for second input vector
                output_dram_addr: DRAM address for output vector
                program_dram_addr: DRAM address for instruction stream. If None, auto-allocated.
                params_dram_addr: DRAM address for parameters. If None, auto-allocated (not used for eltwise).
    
            Returns:
                handler: Callable that takes two input tensors and returns element-wise result in bf16 format
            """
            bytes_per_element = 2
    
            # Verify alignment: size must be a multiple of UE_VECTOR_SIZE (64)
            if size % UE_VECTOR_SIZE != 0:
                raise ValueError(f"Vector size {size} must be a multiple of {UE_VECTOR_SIZE}")
    
            # Validate operation type
            if eltwise_type not in (UE_MODE.ELTWISE_ADD, UE_MODE.ELTWISE_MUL):
                raise ValueError(f"Invalid element-wise operation type: {eltwise_type}. Use ELTWISE_ADD or ELTWISE_MUL")
    
            vector_bytes = size * bytes_per_element
            row_size = size // UE_VECTOR_SIZE
    
            chunk_bytes = min(URAM_NEAR_FULL_SIZE, vector_bytes)
    
            # Start instruction capture
            self.start_capture()
            inst_id = 0
    
            remaining_bytes = vector_bytes
            a_dram_addr = input_a_dram_addr
            b_dram_addr = input_b_dram_addr
            c_dram_addr = output_dram_addr
            while remaining_bytes > 0:
                chunk_bytes = min(remaining_bytes, chunk_bytes)
                row_size = chunk_bytes // (bytes_per_element * UE_VECTOR_SIZE)
    
                # Copy first input vector from DRAM to URAM_A
                self.ue_memcpy_from_dram(a_dram_addr, chunk_bytes, 0, URAM_START_ADDR, URAM_SECTION.URAM_A.value, inst_id)
                inst_id += 1
                self.wait_queue()
    
                # Copy second input vector from DRAM to URAM_B
                self.ue_memcpy_from_dram(b_dram_addr, chunk_bytes, 0, URAM_START_ADDR, URAM_SECTION.URAM_B.value, inst_id)
                inst_id += 1
                self.wait_queue()
    
                # Perform element-wise operation
                self.start_queue(
                    0,  # broadcast_mode (not used)
                    0,  # max_clear_en (not used)
                    1,  # stride_z
                    LALU_MODE.BYPASS.value,  # lalu_mode
                    0,  # scalar (not used)
                    0,  # uram_bram (URAM)
                    URAM_SECTION.URAM_A.value,  # uram_section (write to URAM_A)
                    0,  # uram_dst_addr
                    0,  # dram_to_uram_cpy_start
                    URAM_START_ADDR,  # uram_wb_addr
                    URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                    eltwise_type,  # mode
                    0,  # data_type (not used)
                    URAM_START_ADDR,  # uram_a_start_addr
                    URAM_START_ADDR,  # uram_b_start_addr
                    row_size,  # uram_length
                    0,  # dma_start_addr (not used)
                    0,  # dma_length (not used)
                    0,  # output_size (not used)
                    inst_id  # inst_id
                )
                inst_id += 1
                self.wait_queue()
    
                # Copy result from URAM to DRAM
                self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, URAM_START_ADDR, c_dram_addr, chunk_bytes, inst_id)
                self.wait_queue()
                inst_id += 1
    
                # Update addresses for next chunk
                a_dram_addr += chunk_bytes
                b_dram_addr += chunk_bytes
                c_dram_addr += chunk_bytes
                remaining_bytes -= chunk_bytes
    
            # Finish capture and write instruction stream to DRAM
            self.stop_capture()
            self.generate_instruction_halt()
    
            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
                self.allocate_program_dram(program_size_bytes)
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
                self.write_captured_instructions_to_dram(program_dram_addr)
    
            def handler(input_a, input_b):
                """
                Run element-wise operation using the captured instruction stream.
    
                Supports DeviceTensor for DMA cache optimization.
    
                Args:
                    input_a: First input tensor - torch.Tensor or DeviceTensor
                    input_b: Second input tensor - torch.Tensor or DeviceTensor
    
                Returns:
                    DeviceTensor with result, lazy C2H fetch
                """
                # Extract tensor data and check if DMA can be skipped
                is_a_device = isinstance(input_a, DeviceTensor)
                is_b_device = isinstance(input_b, DeviceTensor)
    
                if is_a_device:
                    data_a = input_a._data
                    shape_a = data_a.shape
                    skip_a = not input_a.needs_dma(input_a_dram_addr)
                else:
                    data_a = input_a
                    shape_a = input_a.shape
                    skip_a = False
    
                if is_b_device:
                    data_b = input_b._data
                    skip_b = not input_b.needs_dma(input_b_dram_addr)
                else:
                    data_b = input_b
                    skip_b = False
    
                # Validate inputs
                assert data_a.dtype == torch.bfloat16, "Input A must be in bf16 format"
                assert data_b.dtype == torch.bfloat16, "Input B must be in bf16 format"
                if data_a.numel() != size:
                    raise ValueError(f"Input A has {data_a.numel()} elements, expected {size}")
                if data_b.numel() != size:
                    raise ValueError(f"Input B has {data_b.numel()} elements, expected {size}")
    
                # Write inputs to DRAM (skip if DeviceTensor already synced)
                if skip_a:
                    print(f"[DMA cache hit] Skipping input A DMA to {hex(input_a_dram_addr)}")
                else:
                    if is_a_device:
                        input_a.sync(input_a_dram_addr)
                    else:
                        self.dma_write(DMA_DEVICE_H2C, input_a_dram_addr, data_a, vector_bytes)
    
                if skip_b:
                    print(f"[DMA cache hit] Skipping input B DMA to {hex(input_b_dram_addr)}")
                else:
                    if is_b_device:
                        input_b.sync(input_b_dram_addr)
                    else:
                        self.dma_write(DMA_DEVICE_H2C, input_b_dram_addr, data_b, vector_bytes)
    
                # Start executing from DRAM and wait for completion
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
    
                self.report_timing_and_instruction_count()
                # 1 FLOP per element for add/mul
                total_flops = size
                print(f"[eltwise_op] {self.report_latency_in_us():.3f} us, {self.report_flop_rate_gflops(total_flops):.3f} GFLOPS")
    
                # Return DeviceTensor if any input was DeviceTensor, else return raw tensor
                output_tensor = DeviceTensor(shape_a, ue=self, dram_addr=output_dram_addr)
                if is_a_device or is_b_device:
                    return output_tensor
                else:
                    return output_tensor.data
    
            return handler
    
        def eltwise_add(self, size: int,
                        input_a_dram_addr: int,
                        input_b_dram_addr: int,
                        output_dram_addr: int,
                        program_dram_addr: Optional[int] = None) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
            """
            Create a reusable element-wise addition operation.
    
            Convenience wrapper around eltwise_op for addition.
    
            Args:
                size: Number of elements in input vectors (must be multiple of 64)
                input_a_dram_addr: DRAM address for first input vector
                input_b_dram_addr: DRAM address for second input vector
                output_dram_addr: DRAM address for output vector
                program_dram_addr: DRAM address for instruction stream. If None, auto-allocated.
    
            Returns:
                handler: Callable that takes two input tensors and returns their sum in bf16 format
            """
            return self.eltwise_op(size, UE_MODE.ELTWISE_ADD, input_a_dram_addr, input_b_dram_addr,
                                   output_dram_addr, program_dram_addr)
    
        def eltwise_mul(self, size: int,
                        input_a_dram_addr: int,
                        input_b_dram_addr: int,
                        output_dram_addr: int,
                        program_dram_addr: Optional[int] = None) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
            """
            Create a reusable element-wise multiplication operation.
    
            Convenience wrapper around eltwise_op for multiplication.
    
            Args:
                size: Number of elements in input vectors (must be multiple of 64)
                input_a_dram_addr: DRAM address for first input vector
                input_b_dram_addr: DRAM address for second input vector
                output_dram_addr: DRAM address for output vector
                program_dram_addr: DRAM address for instruction stream. If None, auto-allocated.
    
            Returns:
                handler: Callable that takes two input tensors and returns their product in bf16 format
            """
            return self.eltwise_op(size, UE_MODE.ELTWISE_MUL, input_a_dram_addr, input_b_dram_addr,
                                   output_dram_addr, program_dram_addr)
    
        def broadcast_op(self, size: int,
                         broadcast_type: UE_MODE,
                         scalar: float,
                         input_dram_addr: int,
                         output_dram_addr: int,
                         program_dram_addr: Optional[int] = None,
                         params_dram_addr: Optional[int] = None) -> Callable[[torch.Tensor], torch.Tensor]:
            """
            Create a reusable broadcast operation with instruction capture.
    
            This method sets up a broadcast operation (add or multiply) that applies a scalar
            value to all elements of an input vector. Instructions are captured once
            during setup and reused for subsequent calls.
    
            URAM Memory Map:
            ================
            URAM_A:
            ┌─────────────────────┐
            │ 0x000               │
            │ input [row_size]    │ ◄─Read
            │                     │
            ├─────────────────────┤
            │ result [row_size]   │ ◄─Write
            │ (overwrites input)  │
            └─────────────────────┘
    
            Data Flow:
              DRAM → URAM_A[0x000] (input)
              Execute: URAM_A[0x000] ⊕ scalar → URAM_A[0x000]
              URAM_A[0x000] → DRAM (result)
    
            Args:
                size: Number of elements in input vector (must be multiple of 64)
                broadcast_type: Operation type (UE_MODE.MUL_BROADCAST or UE_MODE.ADD_BROADCAST)
                scalar: Scalar value to broadcast (float)
                input_dram_addr: DRAM address for input vector
                output_dram_addr: DRAM address for output vector
                program_dram_addr: DRAM address for instruction stream. If None, auto-allocated.
                params_dram_addr: DRAM address for parameters. If None, auto-allocated (not used for broadcast).
    
            Returns:
                handler: Callable that takes input tensor and returns broadcast result in bf16 format
            """
            bytes_per_element = 2
    
            # Verify alignment: size must be a multiple of UE_VECTOR_SIZE (64)
            if size % UE_VECTOR_SIZE != 0:
                raise ValueError(f"Vector size {size} must be a multiple of {UE_VECTOR_SIZE}")
    
            # Validate operation type
            if broadcast_type not in (UE_MODE.MUL_BROADCAST, UE_MODE.ADD_BROADCAST):
                raise ValueError(f"Invalid broadcast operation type: {broadcast_type}. Use MUL_BROADCAST or ADD_BROADCAST")
    
            vector_bytes = size * bytes_per_element
            row_size = size // UE_VECTOR_SIZE
    
            chunk_bytes = min(URAM_NEAR_FULL_SIZE, vector_bytes)
    
            # Convert scalar to bf16 format
            scalar_bf16 = self.float_to_bf16(scalar)
    
            # Auto-allocate params_dram_addr if not provided (not really used for broadcast, but keep for consistency)
            use_auto_mem_management_for_params = False
            if params_dram_addr is None:
                use_auto_mem_management_for_params = True
                params_dram_addr = self.get_params_dram_addr()
            else:
                print(f"Using provided params DRAM address: {params_dram_addr}")
    
            # No params needed for basic broadcast ops, but allocate minimal space for consistency
            if use_auto_mem_management_for_params:
                self.allocate_params_dram(0)  # No params needed
    
            # Start instruction capture
            self.start_capture()
            inst_id = 0
    
            remaining_bytes = vector_bytes
            in_dram_addr = input_dram_addr
            out_dram_addr = output_dram_addr
            while remaining_bytes > 0:
                chunk_bytes = min(remaining_bytes, chunk_bytes)
                row_size = chunk_bytes // (bytes_per_element * UE_VECTOR_SIZE)
    
                # Copy input vector from DRAM to URAM_A
                self.ue_memcpy_from_dram(in_dram_addr, chunk_bytes, 0, URAM_START_ADDR, URAM_SECTION.URAM_A.value, inst_id)
                inst_id += 1
                self.wait_queue()
    
                # Perform broadcast operation with scalar from register
                self.start_queue(
                    BROADCAST_MODE.SCALAR_IN_REG.value,  # broadcast_mode: use scalar from register
                    0,  # max_clear_en
                    1,  # stride_z
                    LALU_MODE.BYPASS.value,  # lalu_mode (not used for broadcast)
                    scalar_bf16, # scalar in bf16 format only for broadcast, rest is bf19 format
                    0,  # uram_bram (URAM)
                    URAM_SECTION.URAM_A.value,  # uram_section (write to URAM_A)
                    0,  # uram_dst_addr (not used for broadcast)
                    0,  # dram_to_uram_cpy_start (not used for broadcast)
                    URAM_START_ADDR,  # uram_wb_addr
                    URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                    broadcast_type,  # mode (MUL_BROADCAST or ADD_BROADCAST)
                    0,  # data_type (not used)
                    URAM_START_ADDR,  # uram_a_start_addr
                    0,  # uram_b_start_addr (not used for broadcast)
                    row_size,  # uram_length
                    0,  # dma_start_addr (not used)
                    0,  # dma_length (not used)
                    0,  # output_size (not used)
                    inst_id  # inst_id
                )
                inst_id += 1
                self.wait_queue()
    
                # Copy result from URAM to DRAM
                self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, URAM_START_ADDR, out_dram_addr, chunk_bytes, inst_id)
                self.wait_queue()
                inst_id += 1
    
                # Update addresses for next chunk
                in_dram_addr += chunk_bytes
                out_dram_addr += chunk_bytes
                remaining_bytes -= chunk_bytes
    
            # Finish capture and write instruction stream to DRAM
            self.stop_capture()
            self.generate_instruction_halt()
    
            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
                self.allocate_program_dram(program_size_bytes)
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
                self.write_captured_instructions_to_dram(program_dram_addr)
    
            op_name = "broadcast_mul" if broadcast_type == UE_MODE.MUL_BROADCAST else "broadcast_add"
    
            def handler(input_tensor):
                """
                Run broadcast operation using the captured instruction stream.
    
                Supports DeviceTensor for DMA cache optimization.
    
                Args:
                    input_tensor: Input tensor - torch.Tensor or DeviceTensor
    
                Returns:
                    DeviceTensor with result, lazy C2H fetch
                """
                # Extract tensor data and check if DMA can be skipped
                if isinstance(input_tensor, DeviceTensor):
                    skip_dma = not input_tensor.needs_dma(input_dram_addr)
                    input_data = input_tensor._data
                else:
                    input_data = input_tensor
                    input_shape = input_tensor.shape
                    skip_dma = False
    
                # Validate input
                if not skip_dma:
                    assert input_data.dtype == torch.bfloat16, "Input must be in bf16 format"
                    if input_data.numel() != size:
                        raise ValueError(f"Input has {input_data.numel()} elements, expected {size}")
    
                # Write input to DRAM (skip if DeviceTensor already synced)
                if skip_dma:
                    print(f"[DMA cache hit] Skipping input DMA to {hex(input_dram_addr)}")
                else:
                    if isinstance(input_tensor, DeviceTensor):
                        input_tensor.sync(input_dram_addr)
                    else:
                        self.dma_write(DMA_DEVICE_H2C, input_dram_addr, input_data, vector_bytes)
    
                # Start executing from DRAM and wait for completion
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
                self.report_timing_and_instruction_count()
                # 1 FLOP per element for add/mul
                total_flops = size
                print(f"[{op_name}] {self.report_latency_in_us():.3f} us, {self.report_flop_rate_gflops(total_flops):.3f} GFLOPS")
    
                # Return DeviceTensor if input was DeviceTensor, else return raw tensor
                if isinstance(input_tensor, DeviceTensor):
                    output_tensor = DeviceTensor(input_tensor._shape, ue=self, dram_addr=output_dram_addr)
                    return output_tensor
                else:
                    output_tensor = DeviceTensor(input_tensor.shape, ue=self, dram_addr=output_dram_addr)
                    return output_tensor.data
    
            return handler
    
        def rms_norm(self, shape: Tuple[int, int],
                     input_dram_addr: int,
                     output_dram_addr: int,
                     program_dram_addr: Optional[int] = None,
                     params_dram_addr: Optional[int] = None,
                     gamma: torch.Tensor = None,
                     beta: torch.Tensor = None) -> Callable[[torch.Tensor], torch.Tensor]:
            """
            Perform RMS Normalization (without mean subtraction) on device
    
            This method performs RMS normalization:
            1) Do the RMS normalization for x: y = x/rms[x]
            2) Optionally multiply by gamma: y = y * gamma
            3) Optionally add beta: y = y + beta
    
            Note: This differs from layer_norm by excluding the mean subtraction step.
            RMS norm normalizes based on the root mean square of the input directly,
            without first subtracting the mean.
    
            URAM Memory Map (row_size = norm_dim/64):
            ==========================================
            URAM_A:                                  URAM_B:
            ┌──────────────────────────────┐         ┌─────────────────────────┐
            │ 0x000                        │         │ 0x000                   │
            │ x[0] (1st vector) [row_size] │         │ gamma (γ) [row_size]    │ ◄─Const
            ├──────────────────────────────┤         ├─────────────────────────┤
            │ x[1] (2nd vector)            │         │ beta (β) [row_size]     │ ◄─Const
            ├──────────────────────────────┤         │ (only if provided)      │
            │ ...                          │         └─────────────────────────┘
            ├──────────────────────────────┤
            │ (chunk limit ~0xF00)         │
            └──────────────────────────────┘
    
            After normalization, results overwrite inputs in URAM_A:
            │ y[0] = x[0]/rms * γ + β      │
            │ y[1] = x[1]/rms * γ + β      │
    
            Per-Vector Computation Steps:
              1. RMS            : rms(x[i])          → LALU (1/rms)
              2. MUL_BROADCAST  : x[i] * (1/rms)     → A[i] (normalized)
              3. ELTWISE_MUL    : norm * γ           → A[i] (if gamma)
              4. ELTWISE_ADD    : result + β         → A[i] (if beta)
    
            Args:
                shape: Tuple of (batch_size, norm_dim) specifying the input shape
                input_dram_addr: DRAM address for input data
                output_dram_addr: DRAM address for output data
                program_dram_addr: DRAM address for instruction stream. If None, auto-allocated.
                params_dram_addr: DRAM address for parameters (gamma, beta). If None, auto-allocated.
                gamma: Optional gamma (scale) parameter vector in bf16 format. If None, gamma is not applied.
                beta: Optional beta (shift) parameter vector in bf16 format. If None, beta is not applied.
    
            Returns:
                handler: Callable that takes input tensor and returns RMS normalized result in bf16 format
            """
            norm_dim = shape[1]
            bytes_per_element = 2
    
            if (norm_dim * bytes_per_element) > URAM_HALF_WAY_SIZE:
                raise ValueError(f"Norm dimension {norm_dim} must be less than {URAM_HALF_WAY_SIZE}")
    
            size = shape[0] * shape[1]
    
            # Verify alignment: size must be a multiple of UE_VECTOR_SIZE (64)
            if norm_dim % UE_VECTOR_SIZE != 0:
                raise ValueError(f"Vector size {norm_dim} must be a multiple of {UE_VECTOR_SIZE}")
    
            # Validate gamma and beta if provided
            if gamma is not None:
                assert gamma.dtype == torch.bfloat16, "Gamma must be in bf16 format"
                assert gamma.numel() == norm_dim, f"Gamma size {gamma.numel()} must match input size {norm_dim}"
            if beta is not None:
                assert beta.dtype == torch.bfloat16, "Beta must be in bf16 format"
                assert beta.numel() == norm_dim, f"Beta size {beta.numel()} must match input size {norm_dim}"
    
            vector_bytes = norm_dim * bytes_per_element
            row_size = norm_dim // UE_VECTOR_SIZE  # Number of 64-element rows
            total_bytes = size * bytes_per_element
    
            # Auto-allocate params_dram_addr if not provided
            use_auto_mem_management_for_params = False
            if params_dram_addr is None:
                use_auto_mem_management_for_params = True
                params_dram_addr = self.get_params_dram_addr()
            else:
                print(f"Using provided params DRAM address: {params_dram_addr}")
    
            # Secondary memory operations (gamma and beta only, no zeros needed for RMS norm)
            size_for_secondary_memory_transfers = 0
    
            gamma_dram_addr = None
            if gamma is not None:
                # Write gamma to DRAM
                gamma_dram_addr = params_dram_addr
                self.dma_write(DMA_DEVICE_H2C, gamma_dram_addr, gamma, vector_bytes)
                size_for_secondary_memory_transfers += vector_bytes
    
            beta_dram_addr = None
            if beta is not None:
                # Write beta to DRAM
                if gamma is None:
                    beta_dram_addr = params_dram_addr
                else:
                    beta_dram_addr = gamma_dram_addr + vector_bytes
    
                self.dma_write(DMA_DEVICE_H2C, beta_dram_addr, beta, vector_bytes)
                size_for_secondary_memory_transfers += vector_bytes
    
            if use_auto_mem_management_for_params:
                self.allocate_params_dram(size_for_secondary_memory_transfers)
    
            result_dram_addr = output_dram_addr
            in_dram_addr = input_dram_addr
            uram_a_start_addr = URAM_START_ADDR
            uram_b_start_addr = URAM_START_ADDR
    
            self.start_capture()
            inst_id = 0
    
            # Copy gamma and beta to URAM_B if they exist
            if size_for_secondary_memory_transfers > 0:
                self.ue_memcpy_from_dram(params_dram_addr, size_for_secondary_memory_transfers, 0, uram_b_start_addr, URAM_SECTION.URAM_B.value, inst_id)
                inst_id += 1
                self.wait_queue()
    
            remaining_bytes = total_bytes
            while remaining_bytes > 0:
                chunk_bytes = min(remaining_bytes, URAM_0XF00_SIZE)
                # Align chunk_bytes to norm_dim * bytes_per_element
                chunk_bytes = (chunk_bytes // (norm_dim * bytes_per_element)) * norm_dim * bytes_per_element
    
                remaining_bytes -= chunk_bytes
                # Copy input vector from DRAM to URAM_A
    
                self.ue_memcpy_from_dram(in_dram_addr, chunk_bytes, 0, uram_a_start_addr, URAM_SECTION.URAM_A.value, inst_id)
                inst_id += 1
                self.wait_queue()
    
                in_dram_addr += chunk_bytes
                number_of_norms = chunk_bytes // (norm_dim * bytes_per_element)
    
                for i in range(number_of_norms):
                    # Step 1: Calculate RMS of x (no mean subtraction)
                    # RMS normalization: 1/sqrt(sum(x^2)/n) is cached in LALU
                    sqrt_n_vector = math.sqrt(norm_dim)
                    scalar_rms = self.float_to_bf19(sqrt_n_vector)
                    self.start_queue(
                        0,  # broadcast_mode
                        0,  # clear_max_en
                        1,  # stride_z
                        LALU_MODE.MODE_RSQRT.value,  # lalu_mode
                        scalar_rms,  # BF19 scalar (sqrt(norm_dim))
                        0,  # uram_bram (URAM = 0)
                        URAM_SECTION.URAM_A.value,  # uram_section
                        0,  # uram_dst_addr
                        0,  # dram_to_uram_cpy_start
                        0,  # uram_wb_addr (no writeback, result in LALU)
                        URAM_WRITE_SRC.URAM_WB_DISABLE.value,  # uram_write_src
                        UE_MODE.RMS,  # mode
                        0,  # data_type not used
                        uram_a_start_addr,  # uram_a_start_addr (input vector)
                        0,  # uram_b_start_addr, not used in RMS
                        row_size,  # uram_length
                        0,  # dma_start_addr
                        0,  # dma_length
                        0,  # output_size
                        inst_id   # inst_id
                    )
                    self.wait_queue()
                    inst_id += 1
    
                    # Step 2: Broadcast multiply to normalize
                    # Multiply x by 1/rms[x] using cached value from LALU
                    self.start_queue(
                        BROADCAST_MODE.LALU_RESULT.value,  # broadcast_mode
                        0,  # clear_max_en
                        1,  # stride_z
                        LALU_MODE.BYPASS.value,  # lalu_mode
                        0,  # scalar (not used)
                        0,  # uram_bram (URAM = 0)
                        URAM_SECTION.URAM_A.value,  # uram_section
                        0,  # uram_dst_addr
                        0,  # dram_to_uram_cpy_start
                        uram_a_start_addr,  # uram_wb_addr (write final result here)
                        URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                        UE_MODE.MUL_BROADCAST,  # mode
                        0,  # data_type not used
                        uram_a_start_addr,  # uram_a_start_addr (input vector)
                        0,  # uram_b_start_addr not used in broadcast multiply
                        row_size,  # uram_length
                        0,  # dma_start_addr
                        0,  # dma_length
                        0,  # output_size
                        inst_id   # inst_id
                    )
                    self.wait_queue()
                    inst_id += 1
    
                    # Step 3 (optional): Multiply by gamma if provided
                    if gamma is not None:
                        # Element-wise multiply: result = result * gamma
                        self.start_queue(
                            0,  # broadcast_mode
                            0,  # clear_max_en
                            1,  # stride_z
                            LALU_MODE.BYPASS.value,  # lalu_mode
                            0,  # scalar (not used)
                            0,  # uram_bram (URAM = 0)
                            URAM_SECTION.URAM_A.value,  # uram_section
                            0,  # uram_dst_addr
                            0,  # dram_to_uram_cpy_start
                            uram_a_start_addr,  # uram_wb_addr (write result back)
                            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                            UE_MODE.ELTWISE_MUL,  # mode
                            0,  # data_type not used
                            uram_a_start_addr,  # uram_a_start_addr (normalized result)
                            uram_b_start_addr,  # uram_b_start_addr (gamma at start)
                            row_size,  # uram_length
                            0,  # dma_start_addr
                            0,  # dma_length
                            0,  # output_size
                            inst_id  # inst_id
                        )
                        self.wait_queue()
                        inst_id += 1
    
                    # Step 4 (optional): Add beta if provided
                    if beta is not None:
                        # Element-wise add: result = result + beta
                        # Beta offset: 0 if gamma is None, row_size if gamma exists
                        beta_uram_offset = 0 if gamma is None else row_size
                        self.start_queue(
                            0,  # broadcast_mode
                            0,  # clear_max_en
                            1,  # stride_z
                            LALU_MODE.BYPASS.value,  # lalu_mode
                            0,  # scalar (not used)
                            0,  # uram_bram (URAM = 0)
                            URAM_SECTION.URAM_A.value,  # uram_section
                            0,  # uram_dst_addr
                            0,  # dram_to_uram_cpy_start
                            uram_a_start_addr,  # uram_wb_addr (write result back)
                            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                            UE_MODE.ELTWISE_ADD,  # mode
                            0,  # data_type not used
                            uram_a_start_addr,  # uram_a_start_addr (result so far)
                            uram_b_start_addr + beta_uram_offset,  # uram_b_start_addr (beta)
                            row_size,  # uram_length
                            0,  # dma_start_addr
                            0,  # dma_length
                            0,  # output_size
                            inst_id  # inst_id
                        )
                        self.wait_queue()
                        inst_id += 1
    
                    uram_a_start_addr += row_size
    
                uram_a_start_addr = URAM_START_ADDR
                self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, uram_a_start_addr, result_dram_addr, chunk_bytes, inst_id)
                self.wait_queue()
                result_dram_addr += chunk_bytes
    
            # Finish capture and write instruction stream to DRAM once
            self.stop_capture()
            self.generate_instruction_halt()
    
            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
                self.allocate_program_dram(program_size_bytes)
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
                self.write_captured_instructions_to_dram(program_dram_addr)
    
            def handler(input_tensor):
                """
                Run RMS normalization using the captured instruction stream.
    
                Supports DeviceTensor for DMA cache optimization.
    
                Args:
                    input_tensor: Input tensor - torch.Tensor or DeviceTensor
                """
                # Extract tensor data and check if DMA can be skipped
                if isinstance(input_tensor, DeviceTensor):
                    input_data = input_tensor._data
                    input_shape = input_data.shape
                    skip_dma = not input_tensor.needs_dma(input_dram_addr)
                else:
                    input_data = input_tensor
                    input_shape = input_tensor.shape
                    skip_dma = False
    
                # Validate input
                assert input_data.dtype == torch.bfloat16, "Input tensor must be in bf16 format"
                if input_data.numel() != size:
                    raise ValueError(f"Input tensor has {input_data.numel()} elements, expected {size}")
    
                # Write input to DRAM (skip if DeviceTensor already synced)
                if skip_dma:
                    print(f"[DMA cache hit] Skipping input DMA to {hex(input_dram_addr)}")
                else:
                    if isinstance(input_tensor, DeviceTensor):
                        input_tensor.sync(input_dram_addr)
                    else:
                        self.dma_write(DMA_DEVICE_H2C, input_dram_addr, input_data, total_bytes)
    
                # Start executing from DRAM and wait for completion
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
                self.report_timing_and_instruction_count()
                # RMS norm: 3 ops per element (square, sum, sqrt) + gamma/beta scaling
                total_flops = 3 * size + (size if gamma is not None else 0) + (size if beta is not None else 0)
                print(f"[rms_norm] {self.report_latency_in_us():.3f} us, {self.report_flop_rate_gflops(total_flops):.3f} GFLOPS")
    
                # Return DeviceTensor if input was DeviceTensor, else return raw tensor
                output_tensor = DeviceTensor(input_shape, ue=self, dram_addr=output_dram_addr)
                if isinstance(input_tensor, DeviceTensor):
                    return output_tensor
                else:
                    return output_tensor.data
    
            return handler
    
        def quantize_weight_to_dram(self,
                                    weight: torch.Tensor,
                                    N: int,
                                    K: int,
                                    block_size: int = 64,
                                    data_type: TYPE = TYPE.INT4,
                                    params_dram_addr: Optional[int] = None) -> Tuple[int, int]:
            """
            Quantize a bf16 weight matrix (N, K) with absmax quantization, pack as INT4/FP4/INT8,
            write to DRAM, and optionally advance params allocator.
    
            Returns:
                (params_dram_addr, params_size): DRAM base address used and total bytes written.
            """
            if N % UE_VECTOR_SIZE != 0:
                raise ValueError(f"N={N} must be a multiple of {UE_VECTOR_SIZE}")
            if K % UE_VECTOR_SIZE != 0:
                raise ValueError(f"K={K} must be a multiple of {UE_VECTOR_SIZE}")
            assert weight.dim() == 2, "Weight must be 2D"
            assert weight.dtype == torch.bfloat16, "Weight must be bfloat16"
            assert weight.shape[0] == N and weight.shape[1] == K, f"Weight shape {weight.shape} must match ({N}, {K})"
    
            matrix = weight.contiguous()  # (N, K)
            use_auto_mem_management_for_params = False
            if params_dram_addr is None:
                use_auto_mem_management_for_params = True
                params_dram_addr = self.get_params_dram_addr()
            else:
                print(f"Using provided params DRAM address: {params_dram_addr}")
    
            print(f"[quantize_weight_to_dram] Quantizing weight (N={N}, K={K}) with absmax quantization (block size {block_size}) as {data_type.name}...")
            matrix_flat = matrix.flatten()
            num_elements = matrix_flat.numel()
            num_blocks = (num_elements + block_size - 1) // block_size
    
            fp4_values = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                      dtype=torch.bfloat16, device=matrix.device)
            if data_type == TYPE.INT4:
                max_val_bf16 = torch.tensor(7.0, dtype=torch.bfloat16, device=matrix.device)
                clamp_min, clamp_max = -8, 7
            elif data_type == TYPE.FP4:
                max_val_bf16 = torch.tensor(6.0, dtype=torch.bfloat16, device=matrix.device)
            else:  # TYPE.INT8
                max_val_bf16 = torch.tensor(127.0, dtype=torch.bfloat16, device=matrix.device)
                clamp_min, clamp_max = -128, 127
    
            quantized_int8 = torch.zeros(num_elements, dtype=torch.int8, device=matrix.device)
            scales_bf16 = torch.zeros(num_blocks, dtype=torch.bfloat16, device=matrix.device)
            for i in range(num_blocks):
                start = i * block_size
                end = min(start + block_size, num_elements)
                block = matrix_flat[start:end]
                block_bf16 = block.to(torch.bfloat16)
                abs_block = block_bf16.abs()
                max_abs = abs_block.max()
                if float(max_abs.item()) == 0.0:
                    scale_bf16 = torch.tensor(1.0, dtype=torch.bfloat16, device=matrix.device)
                else:
                    scale_bf16 = max_abs / max_val_bf16
                scales_bf16[i] = scale_bf16
                if float(max_abs.item()) == 0.0:
                    q_block = torch.zeros(end - start, dtype=torch.int8, device=matrix.device)
                else:
                    scaled = block_bf16 / scale_bf16
                    if data_type == TYPE.FP4:
                        scaled_expanded = scaled.unsqueeze(-1)
                        fp4_values_expanded = fp4_values.unsqueeze(0)
                        distances = torch.abs(scaled_expanded - fp4_values_expanded)
                        closest_indices = torch.argmin(distances, dim=1)
                        fp4_codes = torch.tensor([
                            0b1111, 0b1110, 0b1101, 0b1100, 0b1011, 0b1010, 0b1001, 0b1000,
                            0b0000, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 0b0110, 0b0111,
                        ], dtype=torch.int8, device=matrix.device)
                        q_block = fp4_codes[closest_indices]
                    else:
                        rounded = torch.round(scaled)
                        clamped = rounded.clamp(clamp_min, clamp_max)
                        q_block = clamped.to(torch.int8)
                quantized_int8[start:end] = q_block
    
            num_packed_bytes = (num_elements + 1) // 2
            packed_int4 = torch.zeros(num_packed_bytes, dtype=torch.uint8, device=matrix.device)
            for i in range(0, num_elements, 2):
                byte_idx = i // 2
                if i + 1 < num_elements:
                    val1 = quantized_int8[i].item()
                    val2 = quantized_int8[i + 1].item()
                    packed_int4[byte_idx] = ((val2 & 0xF) << 4) | (val1 & 0xF)
                else:
                    val1 = quantized_int8[i].item()
                    packed_int4[byte_idx] = val1 & 0xF
    
            matrix_dram_addr = params_dram_addr
            matrix_bytes = num_packed_bytes
            self.dma_write(DMA_DEVICE_H2C, matrix_dram_addr, packed_int4, matrix_bytes)
            scale_dram_addr = matrix_dram_addr + matrix_bytes
            scale_bytes = num_blocks * 2
            self.dma_write(DMA_DEVICE_H2C, scale_dram_addr, scales_bf16, scale_bytes)
            params_size = matrix_bytes + scale_bytes
            if use_auto_mem_management_for_params:
                self.allocate_params_dram(params_size)
            return params_dram_addr, params_size
    
        def quantized_matvec(self, matrix: torch.Tensor,
                       M: int, N: int,
                       vector_dram_addr: int,
                       output_dram_addr: int,
                       program_dram_addr: Optional[int] = None,
                       params_dram_addr: Optional[int] = None,
                       gelu_enable: bool = False,
                       silu_enable: bool = False,
                       data_type: TYPE = TYPE.INT4,
                       block_size: int = 64) -> Callable[[torch.Tensor], torch.Tensor]:
            """
            Create a reusable matrix-vector multiplication operation with instruction capture.
    
            This method sets up a matrix-vector multiply that can be called multiple times
            with different input vectors. The matrix is quantized and stored during setup,
            and instructions are captured once for reuse.
    
            URAM Memory Map:
            ================
            URAM_A:                              BRAM (Scale BRAM):
            ┌─────────────────────────┐          ┌─────────────────────────┐
            │ 0x000                   │          │ Scale factors           │ ◄─Loaded in chunks
            │ x (input vector)        │          │ [BF16, one per block]   │   from DRAM
            │ [N elements BF16]       │          │ SCALE_BRAM_SIZE=16KB    │
            └─────────────────────────┘          └─────────────────────────┘
            ...
            ┌─────────────────────────┐
            │ 0x800 (HALFWAY)         │          DRAM (Params region):
            │ y (output vector)       │ ◄─Result ┌─────────────────────────┐
            │ [M elements, accum]     │          │ Packed quantized weights│
            └─────────────────────────┘          │ (INT4: 2 values/byte)   │
                                                 ├─────────────────────────┤
                                                 │ Scale factors (BF16)    │
                                                 └─────────────────────────┘
    
            Chunked Computation (to fit BRAM):
              For each scale chunk:
                1. Copy scales: DRAM → BRAM
                2. DOT_PRODUCT: quantized_W @ x → URAM_A[0x800+offset]
                   (hardware applies dequantization using BRAM scales)
    
            Args:
                matrix: Weight matrix (M x N) in bf16 format - quantized and stored during setup
                M: Number of rows in matrix (output dimension)
                N: Number of columns in matrix (input dimension, must be multiple of 64)
                vector_dram_addr: DRAM address for input vector
                output_dram_addr: DRAM address for output vector
                program_dram_addr: DRAM address for instruction stream. If None, auto-allocated.
                params_dram_addr: DRAM address for parameters (quantized matrix + scales). If None, auto-allocated.
                gelu_enable: Enable GELU activation after multiply (default: False)
                data_type: Data type for quantization - TYPE.INT4, TYPE.INT8, or TYPE.FP4 (default: TYPE.INT4)
                block_size: Block size for quantization (default: 64)
    
            Returns:
                handler: Callable that takes input vector and returns matrix-vector product in bf16 format
            """
            bytes_per_element = 2
    
            # Verify alignment
            if N % UE_VECTOR_SIZE != 0:
                raise ValueError(f"N={N} must be a multiple of {UE_VECTOR_SIZE}")
    
            assert matrix.dim() == 2, "Matrix must be 2D"
            assert matrix.dtype == torch.bfloat16, "Matrix must be bfloat16"
            assert matrix.shape[0] == M and matrix.shape[1] == N, f"Matrix shape {matrix.shape} must match ({M}, {N})"
    
            vector_bytes = N * bytes_per_element
            result_bytes = M * bytes_per_element
    
            # Auto-allocate params_dram_addr if not provided
            use_auto_mem_management_for_params = False
            if params_dram_addr is None:
                use_auto_mem_management_for_params = True
                params_dram_addr = self.get_params_dram_addr()
            else:
                print(f"Using provided params DRAM address: {params_dram_addr}")
    
            # ===== Quantize matrix and write to DRAM =====
            print(f"Quantizing matrix with absmax quantization (block size {block_size}) and packing as {data_type.name}...")
            matrix_flat = matrix.flatten()
            num_elements = matrix_flat.numel()
            num_blocks = (num_elements + block_size - 1) // block_size
    
            # FP4 discrete values
            fp4_values = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                      dtype=torch.bfloat16, device=matrix.device)
    
            # Set quantization range based on data type
            if data_type == TYPE.INT4:
                max_val_bf16 = torch.tensor(7.0, dtype=torch.bfloat16, device=matrix.device)
                clamp_min, clamp_max = -8, 7
            elif data_type == TYPE.FP4:
                max_val_bf16 = torch.tensor(6.0, dtype=torch.bfloat16, device=matrix.device)
            else:  # TYPE.INT8
                max_val_bf16 = torch.tensor(127.0, dtype=torch.bfloat16, device=matrix.device)
                clamp_min, clamp_max = -128, 127
    
            # Quantize each block using absmax quantization
            quantized_int8 = torch.zeros(num_elements, dtype=torch.int8, device=matrix.device)
            scales_bf16 = torch.zeros(num_blocks, dtype=torch.bfloat16, device=matrix.device)
    
            for i in range(num_blocks):
                start = i * block_size
                end = min(start + block_size, num_elements)
                block = matrix_flat[start:end]
    
                block_bf16 = block.to(torch.bfloat16)
                abs_block = block_bf16.abs()
                max_abs = abs_block.max()
    
                if float(max_abs.item()) == 0.0:
                    scale_bf16 = torch.tensor(1.0, dtype=torch.bfloat16, device=matrix.device)
                else:
                    scale_bf16 = max_abs / max_val_bf16
    
                scales_bf16[i] = scale_bf16
    
                if float(max_abs.item()) == 0.0:
                    q_block = torch.zeros(end - start, dtype=torch.int8, device=matrix.device)
                else:
                    scaled = block_bf16 / scale_bf16
    
                    if data_type == TYPE.FP4:
                        scaled_expanded = scaled.unsqueeze(-1)
                        fp4_values_expanded = fp4_values.unsqueeze(0)
                        distances = torch.abs(scaled_expanded - fp4_values_expanded)
                        closest_indices = torch.argmin(distances, dim=1)
                        fp4_codes = torch.tensor([
                            0b1111, 0b1110, 0b1101, 0b1100, 0b1011, 0b1010, 0b1001, 0b1000,
                            0b0000, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 0b0110, 0b0111,
                        ], dtype=torch.int8, device=matrix.device)
                        q_block = fp4_codes[closest_indices]
                    else:
                        rounded = torch.round(scaled)
                        clamped = rounded.clamp(clamp_min, clamp_max)
                        q_block = clamped.to(torch.int8)
    
                quantized_int8[start:end] = q_block
    
            # Pack INT4 values: 2 values per byte
            num_packed_bytes = (num_elements + 1) // 2
            packed_int4 = torch.zeros(num_packed_bytes, dtype=torch.uint8, device=matrix.device)
    
            for i in range(0, num_elements, 2):
                byte_idx = i // 2
                if i + 1 < num_elements:
                    val1 = quantized_int8[i].item()
                    val2 = quantized_int8[i + 1].item()
                    packed_int4[byte_idx] = ((val2 & 0xF) << 4) | (val1 & 0xF)
                else:
                    val1 = quantized_int8[i].item()
                    packed_int4[byte_idx] = val1 & 0xF
    
            # Write quantized matrix to DRAM (params region)
            matrix_dram_addr = params_dram_addr
            matrix_bytes = num_packed_bytes
            self.dma_write(DMA_DEVICE_H2C, matrix_dram_addr, packed_int4, matrix_bytes)
    
            # Write scales to DRAM
            scale_dram_addr = matrix_dram_addr + matrix_bytes
            scale_bytes = num_blocks * 2
            self.dma_write(DMA_DEVICE_H2C, scale_dram_addr, scales_bf16, scale_bytes)
    
            params_size = matrix_bytes + scale_bytes
            if use_auto_mem_management_for_params:
                self.allocate_params_dram(params_size)
    
            print(f"Quantized matrix: {num_elements} elements -> {num_packed_bytes} packed bytes")
            print(f"Scales: {num_blocks} blocks, {scale_bytes} bytes")
    
            use_dequantization_mode = True
    
            # Qantization is done.
            # ===== Capture instructions =====
            self.start_capture()
            inst_id = 0
    
            # Copy input vector from DRAM to URAM_A
            vector_uram_addr = URAM_START_ADDR
            self.ue_memcpy_from_dram(vector_dram_addr, vector_bytes, 0, vector_uram_addr, URAM_SECTION.URAM_A.value, inst_id)
            inst_id += 1
            self.wait_queue()
    
            # Determine LALU mode
            if gelu_enable:
                lalu_mode = LALU_MODE.GELU.value
                lalu_scalar = 0x9FC00
            elif silu_enable:
                lalu_mode = LALU_MODE.SILU.value
                lalu_scalar = 0x9FC00
            else:
                lalu_mode = LALU_MODE.BYPASS.value
                lalu_scalar = 0
    
            remaining_elements = M * N
    
            scale_addr = scale_dram_addr
            wb_addr = URAM_HALFWAY_ADDR
            dram_addr = matrix_dram_addr
            clear_max = 1
    
            while remaining_elements > 0:
                if use_dequantization_mode is True:
                    SCALE_BRAM_ELEMENTS_PER_CHUNK = URAM_NEAR_FULL_ELEMENTS
                else:
                    SCALE_BRAM_ELEMENTS_PER_CHUNK = SCALE_BRAM_ELEMENTS * UE_VECTOR_SIZE
    
                chunk_size = min(remaining_elements, SCALE_BRAM_ELEMENTS_PER_CHUNK) # per chunk
    
                aligned_chunk_size = (chunk_size // (N * UE_VECTOR_SIZE)) * N * UE_VECTOR_SIZE # TODO N is assumed to be a multiple of UE_VECTOR_SIZE
    
                scale_bram_bytes = (aligned_chunk_size // UE_VECTOR_SIZE) * 2
    
                # Copy scale data from DRAM to BRAM
                self.ue_memcpy_from_dram(scale_addr, scale_bram_bytes , MEMCPY_TYPE.BRAM.value, 0, 0, inst_id)
                inst_id += 1
                self.wait_queue()
    
                # Calculate chunk dimensions
                if data_type == TYPE.INT4 or data_type == TYPE.FP4:
                    chunk_dma_bytes = aligned_chunk_size >> 1
                else:
                    chunk_dma_bytes = aligned_chunk_size
    
                # TODO: N is assumed to be a multiple of UE_VECTOR_SIZE
                M_chunk = aligned_chunk_size // N
    
                if use_dequantization_mode is True:
                    # Start queue for dequantization
                    self.start_queue(
                        0,  # broadcast_mode
                        0,  # clear_max_en
                        1,  # stride_z
                        LALU_MODE.BYPASS.value,  # lalu_mode
                        0,  # lalu_scalar
                        0,  # uram_bram (URAM)
                        URAM_SECTION.URAM_B.value,  # uram_section
                        0,  # uram_dst_addr
                        0,  # dram_to_uram_cpy_start
                        0,  # uram_wb_addr
                        URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                        UE_MODE.DEQUANTIZE,  # mode
                        data_type.value,  # data_type
                        0,  # uram_a_start_addr
                        0,  # uram_b_start_addr
                        aligned_chunk_size // UE_VECTOR_SIZE,  # bram_length (N / 64)
                        dram_addr,  # dma_start_addr
                        chunk_dma_bytes,  # dma_length
                        aligned_chunk_size // UE_VECTOR_SIZE,  # output_size
                        inst_id
                    )
                    inst_id += 1
                    self.wait_queue()
    
                    # BF16 dot product: x_vector @ w.T
                    self.start_queue(
                        0,  # broadcast_mode
                        clear_max,  # clear_max_en
                        1,  # stride_z
                        lalu_mode,  # lalu_mode
                        lalu_scalar,  # lalu_scalar
                        0,  # uram_bram
                        URAM_SECTION.URAM_A.value,  # uram_section
                        0,  # uram_dst_addr not used
                        0,  # dram_to_uram_cpy_start not used
                        wb_addr,  # uram_wb_addr
                        URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                        UE_MODE.BF16_DOT_PRODUCT,  # mode
                        0,  # data_type
                        vector_uram_addr,  # uram_a_start_addr
                        URAM_START_ADDR,  # uram_b_start_addr
                        N // UE_VECTOR_SIZE,  # uram_length
                        0,  # dma_start_addr not used
                        M_chunk * N,  # dma_length
                        M_chunk,  # output_size
                        inst_id,
                        0 # bias_adder_en
                    )
                    inst_id += 1
                    self.wait_queue()
                else:
                    # Start queue for dot product
                    self.start_queue(
                        0,  # broadcast_mode
                        clear_max,  # clear_max_en
                        1,  # stride_z
                        lalu_mode,  # lalu_mode
                        lalu_scalar,  # lalu_scalar
                        0,  # uram_bram (URAM)
                        URAM_SECTION.URAM_A.value,  # uram_section
                        0,  # uram_dst_addr
                        0,  # dram_to_uram_cpy_start
                        wb_addr,  # uram_wb_addr
                        URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                        UE_MODE.DOT_PRODUCT,  # mode
                        data_type.value,  # data_type
                        vector_uram_addr,  # uram_a_start_addr
                        0,  # uram_b_start_addr
                        N >> 6,  # bram_length (N / 64)
                        dram_addr,  # dma_start_addr
                        chunk_dma_bytes,  # dma_length
                        M_chunk,  # output_size
                        inst_id
                    )
                    inst_id += 1
                    self.wait_queue()
    
                # Update addresses for next chunk
                scale_addr += scale_bram_bytes
                remaining_elements -= aligned_chunk_size
                wb_addr += M_chunk // UE_VECTOR_SIZE
                dram_addr += chunk_dma_bytes
                clear_max = 0
    
            # Copy result from URAM to DRAM
            self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, URAM_HALFWAY_ADDR, output_dram_addr, result_bytes, inst_id)
            self.wait_queue()
    
            # Finish capture and write instruction stream to DRAM
            self.stop_capture()
            self.generate_instruction_halt()
    
            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
                self.allocate_program_dram(program_size_bytes)
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
                self.write_captured_instructions_to_dram(program_dram_addr)
    
            def handler(input_vector):
                """
                Run matrix-vector multiply using the captured instruction stream.
    
                Supports DeviceTensor for DMA cache optimization.
    
                Args:
                    input_vector: Input vector (N,) or (N, 1) - torch.Tensor or DeviceTensor
    
                Returns:
                    DeviceTensor with result vector (M,), lazy C2H fetch
                """
                # Extract tensor data and check if DMA can be skipped
                if isinstance(input_vector, DeviceTensor):
                    input_data = input_vector._data
                    skip_dma = not input_vector.needs_dma(vector_dram_addr)
                else:
                    input_data = input_vector
                    skip_dma = False
    
                assert input_data.dtype == torch.bfloat16, "Input vector must be in bf16 format"
                vec_size = input_data.numel()
                if vec_size != N:
                    raise ValueError(f"Input vector has {vec_size} elements, expected {N}")
    
                # Write input vector to DRAM (skip if DeviceTensor already synced)
                if skip_dma:
                    print(f"[DMA cache hit] Skipping input DMA to {hex(vector_dram_addr)}")
                else:
                    if isinstance(input_vector, DeviceTensor):
                        input_vector.sync(vector_dram_addr)
                    else:
                        self.dma_write(DMA_DEVICE_H2C, vector_dram_addr, input_data, vector_bytes)
    
                # Start executing from DRAM and wait for completion
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
                self.report_timing_and_instruction_count()
                # Matrix-vector multiply: 2*M*N FLOPs
                total_flops = 2 * M * N
                print(f"[bf16_matvec] {self.report_latency_in_us():.3f} us, {self.report_flop_rate_gflops(total_flops):.3f} GFLOPS")
    
                # Return DeviceTensor if input was DeviceTensor, else return raw tensor
                output_tensor = DeviceTensor((M,), ue=self, dram_addr=output_dram_addr)
                if isinstance(input_vector, DeviceTensor):
                    return output_tensor
                else:
                    return output_tensor.data
    
            return handler
    
        def matmat_mul_quantized_weights(self, weight: torch.Tensor,
                                M: int, K: int, N: int,
                                input_dram_addr: int,
                                output_dram_addr: int,
                                program_dram_addr: Optional[int] = None,
                                params_dram_addr: Optional[int] = None,
                                bias: Optional[torch.Tensor] = None,
                                gelu_enable: bool = False,
                                silu_enable: bool = False,
                                softmax_enable: bool = False,
                                data_type: TYPE = TYPE.FP4,
                                block_size: int = 64) -> Callable[[torch.Tensor], torch.Tensor]:
            """
            Create a reusable quantized matrix-matrix multiplication with K-partition optimization.
    
            Computes: Y = X @ W + bias [+ GELU/SILU/Softmax]
    
            Where:
                X: Input activation matrix (M x K) in bf16 - provided per call
                W: Weight matrix (N x K) - quantized and stored during setup
                Y: Output matrix (M x N) in bf16
                bias: Optional bias vector (N,) in bf16
    
            Dimension Convention:
            =====================
                M = batch/sequence dimension (number of input vectors)
                K = input features (shared dimension, must be multiple of 64)
                N = output features (must be multiple of 64)
    
            Partitioning Strategy (K-partition optimization):
            =================================================
            Uses dynamic URAM splitting based on bf16_matmat_activation_k_partition:
    
            1. max_N_chunk = based on BRAM scale capacity
               - Scales stored in BRAM, each N_chunk x K block needs scales
    
            2. Dynamic L calculation for optimal input/output balance:
               - ratio = (max_N_chunk / K) + 1.0
               - L = URAM_FULL / ratio
               - max_M_chunk = min(M, L // K, (URAM_FULL - L) // max_N_chunk)
    
            URAM Memory Map:
            ================
            URAM_A:
            ┌─────────────────────────┐
            │ 0x000                   │
            │ A_chunk (max_M_chunk×K) │ ◄─ Input vectors
            │ (first L elements)      │
            ├─────────────────────────┤
            │ L/64 (dynamic)          │
            │ y (1 x N)               │ ◄─ Output (one row, reused)
            └─────────────────────────┘
    
            Processing Flow:
            ================
            For each M_chunk of input rows:
              1. Load A_chunk (max_M_chunk x K) to URAM_A[0x000]
              For each row i in M_chunk:
                For each scale chunk (N_chunk weights):
                  2. Load scales to BRAM
                  3. DOT_PRODUCT: x[i] @ W_chunk → y[N_chunk_range]
                     - Quantized weights streamed from DRAM
                     - Scales from BRAM for dequantization
                4. ELTWISE_ADD: y + bias → y (if bias)
                5. Write row to DRAM at output_addr + i * N
    
            Args:
                weight: Weight matrix (N x K) in bf16 format - quantized and stored during setup
                M: Batch/sequence dimension (number of input vectors)
                K: Input features (must be multiple of 64)
                N: Output features (must be multiple of 64)
                input_dram_addr: DRAM address for input matrix (M x K) - provided per call
                output_dram_addr: DRAM address for output matrix (M x N)
                program_dram_addr: DRAM address for instruction stream. If None, auto-allocated.
                params_dram_addr: DRAM address for quantized weights and scales. If None, auto-allocated.
                bias: Optional bias vector (N,) in bf16 format
                gelu_enable: Enable GELU activation after multiply (default: False)
                silu_enable: Enable SILU activation after multiply (default: False)
                softmax_enable: Enable row-wise softmax after multiply (default: False)
                data_type: Data type for quantization - TYPE.INT4, TYPE.FP4, TYPE.INT8 (default: TYPE.FP4)
                block_size: Block size for quantization (default: 64)
    
            Note:
                - When softmax_enable=True, gelu_enable and silu_enable must be False.
                - If data_type=TYPE.BF16, delegates to bf16_matmat instead.
    
            Returns:
                handler: Callable that takes input (M, K) and returns output (M, N) in bf16 format
            """
            bytes_per_element = 2
    
            # Verify alignment
            if N % UE_VECTOR_SIZE != 0:
                raise ValueError(f"N={N} must be a multiple of {UE_VECTOR_SIZE}")
            if K % UE_VECTOR_SIZE != 0:
                raise ValueError(f"K={K} must be a multiple of {UE_VECTOR_SIZE}")
    
            assert weight.dim() == 2, "Weight must be 2D"
            assert weight.dtype == torch.bfloat16, "Weight must be bfloat16"
            assert weight.shape[0] == N and weight.shape[1] == K, f"Weight shape {weight.shape} must match ({N}, {K})"
    
            if bias is not None:
                assert bias.dim() == 1, "Bias must be 1D"
                assert bias.dtype == torch.bfloat16, "Bias must be bfloat16"
                assert bias.shape[0] == N, f"Bias shape {bias.shape[0]} must match N={N}"
    
            # Validate mutually exclusive options: when softmax is enabled, gelu and silu cannot be enabled
            if softmax_enable:
                if gelu_enable or silu_enable:
                    raise ValueError("When softmax_enable=True, gelu_enable and silu_enable must be False")
            activation_count = sum([gelu_enable, silu_enable, softmax_enable])
            if activation_count > 1:
                raise ValueError("gelu_enable, silu_enable, and softmax_enable are mutually exclusive")
    
            # If BF16 type is requested, delegate to bf16_matmat for consistency
            if data_type == TYPE.BF16:
                return self.bf16_matmat(
                    weight=weight,
                    M=M, K=K, N=N,
                    input_dram_addr=input_dram_addr,
                    output_dram_addr=output_dram_addr,
                    program_dram_addr=program_dram_addr,
                    params_dram_addr=params_dram_addr,
                    bias=bias,
                    gelu_enable=gelu_enable,
                    silu_enable=silu_enable,
                    softmax_enable=softmax_enable
                )
    
            if softmax_enable:
                raise ValueError("Softmax is not supported for quantized_matmat")
    
            op_suffix = ""
            if gelu_enable:
                op_suffix = " + GELU"
            elif silu_enable:
                op_suffix = " + SILU"
            elif softmax_enable:
                op_suffix = " + Softmax"
            print(f"[quantized_matmat{op_suffix}] M={M}, K={K}, N={N}, bias={bias is not None}, data_type={data_type.name} where X is (M, K) and W is (N, K) and Y is (M, N)")
    
            matrix = weight.contiguous()  # (N, K)
    
            # ===== Calculate URAM partitioning using K-partition optimization =====
            # Align BRAM size to K dimension for scale chunks
            ALIGNMENT_SIZE = K * 2  # 2 bytes per bf16 scale
            BRAM_SIZE_ALIGNED = (SCALE_BRAM_SIZE_BYTES // ALIGNMENT_SIZE) * ALIGNMENT_SIZE
    
            # Calculate max_N_chunk based on BRAM scale capacity
            # Each N_chunk x K block needs (N_chunk * K / 64) scales, each scale is 2 bytes
            max_N_chunk = ((BRAM_SIZE_ALIGNED // 2) * UE_VECTOR_SIZE // K // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
            max_N_chunk = min(max_N_chunk, N)
            if max_N_chunk < UE_VECTOR_SIZE:
                max_N_chunk = min(N, UE_VECTOR_SIZE)  # At minimum, process 64 output elements
    
            # Use dynamic L calculation from bf16_matmat_activation_k_partition
            # L = URAM_FULL / ((max_N_chunk / K) + 1.0)
            # This balances input storage (L elements) vs output accumulation space (URAM_FULL - L)
            ratio = (max_N_chunk / K) + 1.0
            L = int(URAM_FULL_ELEMENTS / ratio)
    
            # Calculate max_M_chunk: how many input vectors fit while leaving room for output
            max_M_chunk = min(M, L // K, (URAM_FULL_ELEMENTS - L) // max_N_chunk)
            max_M_chunk = max(1, max_M_chunk)  # At least 1 vector
    
            # Output address starts after input storage
            output_row_uram_addr = L // UE_VECTOR_SIZE
    
            print(f"[quantized_matmat] K-partition optimization: L={L}, max_M_chunk={max_M_chunk}, max_N_chunk={max_N_chunk}")
            print(f"[quantized_matmat] Input storage: {max_M_chunk * K} elements, Output storage: {URAM_FULL_ELEMENTS - L} elements")
    
            # Auto-allocate params_dram_addr if not provided
            use_auto_mem_management_for_params = False
            if params_dram_addr is None:
                use_auto_mem_management_for_params = True
                params_dram_addr = self.get_params_dram_addr()
            else:
                print(f"Using provided params DRAM address: {params_dram_addr}")
    
            # ===== Quantize transposed weight matrix (N, K) and write to DRAM =====
            print(f"[quantized_matmat] Quantizing weight (N={N}, K={K}) with absmax quantization (block size {block_size}) as {data_type.name}...")
            matrix_flat = matrix.flatten()  # matrix is (N, K) transposed
            num_elements = matrix_flat.numel()
            num_blocks = (num_elements + block_size - 1) // block_size
    
            # FP4 discrete values
            fp4_values = torch.tensor([-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
                                      dtype=torch.bfloat16, device=matrix.device)
    
            # Set quantization range based on data type
            if data_type == TYPE.INT4:
                max_val_bf16 = torch.tensor(7.0, dtype=torch.bfloat16, device=matrix.device)
                clamp_min, clamp_max = -8, 7
            elif data_type == TYPE.FP4:
                max_val_bf16 = torch.tensor(6.0, dtype=torch.bfloat16, device=matrix.device)
            else:  # TYPE.INT8
                max_val_bf16 = torch.tensor(127.0, dtype=torch.bfloat16, device=matrix.device)
                clamp_min, clamp_max = -128, 127
    
            # Quantize each block using absmax quantization
            quantized_int8 = torch.zeros(num_elements, dtype=torch.int8, device=matrix.device)
            scales_bf16 = torch.zeros(num_blocks, dtype=torch.bfloat16, device=matrix.device)
    
            for i in range(num_blocks):
                start = i * block_size
                end = min(start + block_size, num_elements)
                block = matrix_flat[start:end]
    
                block_bf16 = block.to(torch.bfloat16)
                abs_block = block_bf16.abs()
                max_abs = abs_block.max()
    
                if float(max_abs.item()) == 0.0:
                    scale_bf16 = torch.tensor(1.0, dtype=torch.bfloat16, device=matrix.device)
                else:
                    scale_bf16 = max_abs / max_val_bf16
    
                scales_bf16[i] = scale_bf16
    
                if float(max_abs.item()) == 0.0:
                    q_block = torch.zeros(end - start, dtype=torch.int8, device=matrix.device)
                else:
                    scaled = block_bf16 / scale_bf16
    
                    if data_type == TYPE.FP4:
                        scaled_expanded = scaled.unsqueeze(-1)
                        fp4_values_expanded = fp4_values.unsqueeze(0)
                        distances = torch.abs(scaled_expanded - fp4_values_expanded)
                        closest_indices = torch.argmin(distances, dim=1)
                        fp4_codes = torch.tensor([
                            0b1111, 0b1110, 0b1101, 0b1100, 0b1011, 0b1010, 0b1001, 0b1000,
                            0b0000, 0b0001, 0b0010, 0b0011, 0b0100, 0b0101, 0b0110, 0b0111,
                        ], dtype=torch.int8, device=matrix.device)
                        q_block = fp4_codes[closest_indices]
                    else:
                        rounded = torch.round(scaled)
                        clamped = rounded.clamp(clamp_min, clamp_max)
                        q_block = clamped.to(torch.int8)
    
                quantized_int8[start:end] = q_block
    
            # Pack INT4/FP4 values: 2 values per byte
            num_packed_bytes = (num_elements + 1) // 2
            packed_int4 = torch.zeros(num_packed_bytes, dtype=torch.uint8, device=matrix.device)
    
            for i in range(0, num_elements, 2):
                byte_idx = i // 2
                if i + 1 < num_elements:
                    val1 = quantized_int8[i].item()
                    val2 = quantized_int8[i + 1].item()
                    packed_int4[byte_idx] = ((val2 & 0xF) << 4) | (val1 & 0xF)
                else:
                    val1 = quantized_int8[i].item()
                    packed_int4[byte_idx] = val1 & 0xF
    
            # Write quantized matrix to DRAM (params region)
            matrix_dram_addr = params_dram_addr
            matrix_bytes = num_packed_bytes
            self.dma_write(DMA_DEVICE_H2C, matrix_dram_addr, packed_int4, matrix_bytes)
    
            # Write scales to DRAM
            scale_dram_addr = matrix_dram_addr + matrix_bytes
            scale_bytes = num_blocks * 2
            self.dma_write(DMA_DEVICE_H2C, scale_dram_addr, scales_bf16, scale_bytes)
    
            # Write bias to DRAM if provided
            bias_dram_addr = None
            if bias is not None:
                bias_dram_addr = scale_dram_addr + scale_bytes
                bias_bytes = N * bytes_per_element
                # Pad bias to multiple of 64
                self.dma_write(DMA_DEVICE_H2C, bias_dram_addr, bias, bias_bytes)
                params_size = matrix_bytes + scale_bytes + bias_bytes
            else:
                params_size = matrix_bytes + scale_bytes
    
            if use_auto_mem_management_for_params:
                self.allocate_params_dram(params_size)
    
            print(f"[quantized_matmat] Quantized weight: {num_elements} elements -> {num_packed_bytes} packed bytes")
            print(f"[quantized_matmat] Scales: {num_blocks} blocks, {scale_bytes} bytes")
            if bias is not None:
                print(f"[quantized_matmat] Bias: {N} elements, {N * 2} bytes")
    
            # ===== Calculate total bytes for M vectors (no padding) =====
            num_batches = (M + max_M_chunk - 1) // max_M_chunk
            total_input_bytes = M * K * bytes_per_element
            total_output_bytes = M * N * bytes_per_element
    
            print(f"[quantized_matmat] Processing M={M} vectors in {num_batches} batches of up to {max_M_chunk}")
    
            # ===== Capture instructions for ALL batches =====
            # The program handles all DRAM↔URAM transfers internally
            # Hardware matrix is (N, K) stored, computing y = W_hw @ x
            self.start_capture()
            inst_id = 0
    
            # Determine LALU mode (only used if gelu_enable or silu_enable, softmax handles its own LALU)
            if gelu_enable:
                lalu_mode = LALU_MODE.GELU.value
                lalu_scalar = 0x9FC00 # bf19 1.0
            elif silu_enable:
                lalu_mode = LALU_MODE.SILU.value
                lalu_scalar = 0
            else:
                lalu_mode = LALU_MODE.BYPASS.value
                lalu_scalar = 0
    
            # Process all batches using while loop with K-partition optimization
            # Like bf16_matmat_activation_k_partition: process row by row, write each row to DRAM
            vectors_remaining = M
            batch_input_dram_addr = input_dram_addr
            batch_output_dram_addr = output_dram_addr
    
            while vectors_remaining > 0:
                # Current batch size: max_M_chunk or remaining vectors
                current_batch_size = min(max_M_chunk, vectors_remaining)
                current_input_bytes = current_batch_size * K * bytes_per_element
    
                # Copy batch of input vectors from DRAM to URAM_A[0x000]
                self.ue_memcpy_from_dram(batch_input_dram_addr, current_input_bytes, 0, URAM_START_ADDR, URAM_SECTION.URAM_A.value, inst_id)
                inst_id += 1
                self.wait_queue()
    
                # Process each vector in this batch
                for p in range(current_batch_size):
                    # Input vector address in URAM: 0x000 + p * (K / 64) rows
                    input_uram_addr = URAM_START_ADDR + p * (K // UE_VECTOR_SIZE)
                    # Output goes to the dynamic output region (L // 64)
                    # Each row is written to DRAM after processing, so we reuse same output location
    
                    scale_remaining_bytes = N * K * 2 // UE_VECTOR_SIZE
    
                    scale_addr = scale_dram_addr
                    wb_addr = output_row_uram_addr  # Use dynamic output address
                    dram_addr = matrix_dram_addr
                    clear_max = 1
    
                    bias_offset = bias_dram_addr
    
                    while scale_remaining_bytes > 0:
                        chunk_scale_bytes = min(scale_remaining_bytes, BRAM_SIZE_ALIGNED)
    
                        number_of_elements = (chunk_scale_bytes * UE_VECTOR_SIZE) >> 1
    
                        # Copy scale data from DRAM to BRAM
                        self.ue_memcpy_from_dram(scale_addr, chunk_scale_bytes, MEMCPY_TYPE.BRAM.value, 0, 0, inst_id)
                        inst_id += 1
                        self.wait_queue()
    
                        # Calculate chunk dimensions
                        if data_type == TYPE.INT4 or data_type == TYPE.FP4:
                            chunk_dma_bytes = number_of_elements >> 1
                        else:  # INT8
                            chunk_dma_bytes = number_of_elements
    
                        N_chunk = number_of_elements // K
    
                        if bias is not None:
                            self.ue_memcpy_from_dram(bias_offset, N_chunk * bytes_per_element, MEMCPY_TYPE.BIAS_BRAM.value, 0, 0, inst_id)
                            self.wait_queue()
                            inst_id += 1
                            bias_offset += N_chunk * bytes_per_element
    
                        # Start queue for dot product
                        self.start_queue(
                            0,  # broadcast_mode
                            clear_max,  # clear_max_en
                            1,  # stride_z
                            lalu_mode,  # lalu_mode
                            lalu_scalar,  # lalu_scalar
                            0,  # uram_bram (URAM)
                            URAM_SECTION.URAM_A.value,  # uram_section
                            0,  # uram_dst_addr
                            0,  # dram_to_uram_cpy_start
                            wb_addr,  # uram_wb_addr
                            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                            UE_MODE.DOT_PRODUCT,  # mode
                            data_type.value,  # data_type
                            input_uram_addr,  # uram_a_start_addr (this vector's input)
                            0,  # uram_b_start_addr (not used for dot product)
                            K >> 6,  # inner dimension of the matrix in 64 element units
                            dram_addr,  # dma_start_addr
                            chunk_dma_bytes,  # dma_length
                            N_chunk,  # output_size
                            inst_id,
                            1 if bias is not None else 0
                        )
                        inst_id += 1
                        self.wait_queue()
    
                        # Update addresses for next chunk
                        scale_addr += chunk_scale_bytes
                        scale_remaining_bytes -= chunk_scale_bytes
                        wb_addr += N_chunk // UE_VECTOR_SIZE
                        dram_addr += chunk_dma_bytes
                        clear_max = 0
    
                    # Write this row to DRAM (K-partition style: row-by-row output)
                    row_output_dram_addr = batch_output_dram_addr + p * N * bytes_per_element
                    self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, output_row_uram_addr, row_output_dram_addr, N * bytes_per_element, inst_id)
                    inst_id += 1
                    self.wait_queue()
    
                # Update for next batch
                vectors_remaining -= current_batch_size
                batch_input_dram_addr += current_input_bytes
                batch_output_dram_addr += current_batch_size * N * bytes_per_element
    
            # Finish capture and write instruction stream to DRAM
            self.stop_capture()
            self.generate_instruction_halt()
    
            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
                self.allocate_program_dram(program_size_bytes)
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
                self.write_captured_instructions_to_dram(program_dram_addr)
    
            def handler(input_matrix) -> torch.Tensor:
                """
                Run matrix-matrix multiply: Y = X @ W + bias
    
                Supports DeviceTensor for DMA cache optimization.
    
                Args:
                    input_matrix: Input (M, K) - torch.Tensor or DeviceTensor
    
                Returns:
                    Result matrix (M, N) in bf16 format
                """
                # Extract tensor data and check if DMA can be skipped
                if isinstance(input_matrix, DeviceTensor):
                    input_data = input_matrix._data
                    skip_dma = not input_matrix.needs_dma(input_dram_addr)
                else:
                    input_data = input_matrix
                    skip_dma = False
    
                assert input_data.dtype == torch.bfloat16, "Input matrix must be in bf16 format"
    
                if input_data.dim() == 1:
                    input_data = input_data.unsqueeze(0)
    
                input_M, input_K = input_data.shape
                if input_M != M:
                    raise ValueError(f"Input matrix has {input_M} rows, expected M={M}")
                if input_K != K:
                    raise ValueError(f"Input matrix has {input_K} columns, expected K={K}")
    
                # DMA input to DRAM (skip if DeviceTensor is already synced)
                if skip_dma:
                    print(f"[DMA cache hit] Skipping input DMA to {hex(input_dram_addr)}")
                else:
                    if isinstance(input_matrix, DeviceTensor):
                        input_matrix.sync(input_dram_addr)
                    else:
                        self.dma_write(DMA_DEVICE_H2C, input_dram_addr, input_data.flatten(), total_input_bytes)
    
                # Start executing from DRAM and wait for completion
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
                self.report_timing_and_instruction_count()
                # Matrix-matrix multiply: 2*M*K*N FLOPs + bias addition + softmax ops
                total_flops = 2 * M * K * N
                if bias is not None:
                    total_flops += M * N * 2
                if softmax_enable:
                    total_flops += M * N * 5  # max(M*N) + sub(M*N) + exp(M*N) + sum(M*N) + div(1) + mul(M*N)
                print(f"[quantized_matmat{op_suffix}] {self.report_latency_in_us():.3f} us, {self.report_flop_rate_gflops(total_flops):.3f} GFLOPS")
    
                # Return DeviceTensor if input was DeviceTensor, else return raw tensor
                output_tensor = DeviceTensor((M, N), ue=self, dram_addr=output_dram_addr)
                if isinstance(input_matrix, DeviceTensor):
                    return output_tensor
                else:
                    return output_tensor.data
    
            return handler
    
        def patching(self,
                    input_dram_addr: int,
                    output_dram_addr: int,
                    program_dram_addr: Optional[int] = None,
                    params_dram_addr: Optional[int] = None,
                    block_size: int = 64) -> Callable[[torch.Tensor], torch.Tensor]:
          
            bytes_per_element = 2
            M = 1
            K = 1024
            N = 64
            data_type = TYPE.INT4
    
            matrix_dram_addr_list = []
            scale_dram_addr_list = []
    
            for matrix_idx in range(16):
                weight = torch.zeros(64, 1024).to(torch.bfloat16)
                for i in range(48):
                    weight[i, matrix_idx*4 + i%4 + (i//4)*64] = 1.0
                params_addr, _ = self.quantize_weight_to_dram(
                    weight, N, K, block_size=block_size, data_type=data_type, params_dram_addr=params_dram_addr
                )
                num_elements = N * K
                matrix_bytes = num_elements // 2
                matrix_dram_addr_list.append(params_addr)
                scale_dram_addr_list.append(params_addr + matrix_bytes)
    
            ALIGNMENT_SIZE = K * 2
            BRAM_SIZE_ALIGNED = (SCALE_BRAM_SIZE_BYTES // ALIGNMENT_SIZE) * ALIGNMENT_SIZE
    
            # ===== Calculate total bytes for M vectors (no padding) =====
            total_input_bytes = 3 * 384 * 384 * bytes_per_element
            #total_output_bytes = 64 * 96 * 96 * bytes_per_element
    
            # Output address starts after input storage
            output_row_uram_addr = URAM_HALFWAY_ADDR
    
            # ===== Capture instructions for ALL batches =====
            # The program handles all DRAM↔URAM transfers internally
            # Hardware matrix is (N, K) stored, computing y = W_hw @ x
            self.start_capture()
            inst_id = 0
    
            lalu_mode = LALU_MODE.BYPASS.value
            lalu_scalar = 0
    
            # Process all batches using while loop with K-partition optimization
            # Like bf16_matmat_activation_k_partition: process row by row, write each row to DRAM
            batch_input_dram_addr = input_dram_addr
            batch_output_dram_addr = output_dram_addr
            patch_offset_in_bytes = 0
    
            for p_i in range(96): # 16 rows per patch
                for p_j in range(96 // 16): # 16 patches per row
                        patch_offset_in_bytes = p_i * 4 * 384 * bytes_per_element + p_j * 64 * bytes_per_element
                        # Current batch size: max_M_chunk or remaining vectors
                        current_batch_size = 16
                        current_input_bytes = 128 # min(128, 32, 4) # TODO
                        # Copy batch of input vectors from DRAM to URAM_A[0x000]
                        for channel in range(3):
                            for row_idx in range(4):
                                # total bytes 12*64*2 = 1536 bytes
                                uram_addr = URAM_START_ADDR + channel * 4 + row_idx
                                batch_input_dram_addr = input_dram_addr + channel * 384 * 384 * bytes_per_element + row_idx * 384 * bytes_per_element + patch_offset_in_bytes
                                self.ue_memcpy_from_dram(batch_input_dram_addr, current_input_bytes, 0, uram_addr, URAM_SECTION.URAM_A.value, inst_id)
                                inst_id += 1
                                self.wait_queue()
    
                        # Process each vector in this batch
                        for p in range(current_batch_size):
                            # THIS PART IS MATRIX VECTOR MULTIPLY
                            # M = 1, K = 512, N = 64
    
                            # Input vector address in URAM: 0x000 + p * (K / 64) rows
                            input_uram_addr = URAM_START_ADDR
                            # Output goes to the dynamic output region (L // 64)
                            # Each row is written to DRAM after processing, so we reuse same output location
    
                            scale_remaining_bytes = N * K * 2 // UE_VECTOR_SIZE
    
                            scale_addr = scale_dram_addr_list[p]
                            wb_addr = output_row_uram_addr + p
                            dram_addr = matrix_dram_addr_list[p]
                            clear_max = 1
    
                            while scale_remaining_bytes > 0:
                                chunk_scale_bytes = min(scale_remaining_bytes, BRAM_SIZE_ALIGNED)
    
                                number_of_elements = (chunk_scale_bytes * UE_VECTOR_SIZE) >> 1
    
                                # Copy scale data from DRAM to BRAM
                                self.ue_memcpy_from_dram(scale_addr, chunk_scale_bytes, MEMCPY_TYPE.BRAM.value, 0, 0, inst_id)
                                inst_id += 1
                                self.wait_queue()
    
                                # Calculate chunk dimensions
                                if data_type == TYPE.INT4 or data_type == TYPE.FP4:
                                    chunk_dma_bytes = number_of_elements >> 1
                                else:  # INT8
                                    chunk_dma_bytes = number_of_elements
    
                                N_chunk = number_of_elements // K
    
                                # Start queue for dot product
                                self.start_queue(
                                    0,  # broadcast_mode
                                    clear_max,  # clear_max_en
                                    1,  # stride_z
                                    lalu_mode,  # lalu_mode
                                    lalu_scalar,  # lalu_scalar
                                    0,  # uram_bram (URAM)
                                    URAM_SECTION.URAM_A.value,  # uram_section
                                    0,  # uram_dst_addr
                                    0,  # dram_to_uram_cpy_start
                                    wb_addr,  # uram_wb_addr
                                    URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                                    UE_MODE.DOT_PRODUCT,  # mode
                                    data_type.value,  # data_type
                                    input_uram_addr,  # uram_a_start_addr (this vector's input)
                                    0,  # uram_b_start_addr (not used for dot product)
                                    K >> 6,  # inner dimension of the matrix in 64 element units
                                    dram_addr,  # dma_start_addr
                                    chunk_dma_bytes,  # dma_length
                                    N_chunk,  # output_size
                                    inst_id,
                                    0 # no bias
                                )
                                inst_id += 1
                                self.wait_queue()
    
                                # Update addresses for next chunk
                                scale_addr += chunk_scale_bytes
                                scale_remaining_bytes -= chunk_scale_bytes
                                wb_addr += N_chunk // UE_VECTOR_SIZE
                                dram_addr += chunk_dma_bytes
                                clear_max = 0
    
                        self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, output_row_uram_addr, batch_output_dram_addr, 16 * N * bytes_per_element, inst_id)
                        inst_id += 1
                        self.wait_queue()
    
                        # # Update for next batch
                        # patch_remaining -= 16
                        # patch_offset_in_bytes += 4 * 384 * bytes_per_element if patch_remaining % 96 == 0 else 1 * 64 * bytes_per_element
                        batch_output_dram_addr += 16 * N * bytes_per_element
    
            # Finish capture and write instruction stream to DRAM
            self.stop_capture()
            self.generate_instruction_halt()
    
            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
                self.allocate_program_dram(program_size_bytes)
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
                self.write_captured_instructions_to_dram(program_dram_addr)
    
            def handler(input_matrix) -> torch.Tensor:
                # Extract tensor data and check if DMA can be skipped
                if isinstance(input_matrix, DeviceTensor):
                    input_data = input_matrix._data
                    skip_dma = not input_matrix.needs_dma(input_dram_addr)
                else:
                    input_data = input_matrix
                    skip_dma = False
    
                assert input_data.dtype == torch.bfloat16, "Input matrix must be in bf16 format"
    
                if input_data.dim() == 1:
                    input_data = input_data.unsqueeze(0)
    
                # DMA input to DRAM (skip if DeviceTensor is already synced)
                if skip_dma:
                    print(f"[DMA cache hit] Skipping input DMA to {hex(input_dram_addr)}")
                else:
                    if isinstance(input_matrix, DeviceTensor):
                        input_matrix.sync(input_dram_addr)
                    else:
                        self.dma_write(DMA_DEVICE_H2C, input_dram_addr, input_data.flatten(), total_input_bytes)
    
                # Start executing from DRAM and wait for completion
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
                self.report_timing_and_instruction_count()
    
                # Return DeviceTensor if input was DeviceTensor, else return raw tensor
                output_tensor = DeviceTensor((96*96, 64), ue=self, dram_addr=output_dram_addr)
                if isinstance(input_matrix, DeviceTensor):
                    return output_tensor
                else:
                    return output_tensor.data
    
            return handler
    
        def bf16_matmat(self, weight: torch.Tensor,
                            M: int, K: int, N: int,
                            input_dram_addr: int,
                            output_dram_addr: int,
                            program_dram_addr: Optional[int] = None,
                            params_dram_addr: Optional[int] = None,
                            bias: Optional[torch.Tensor] = None,
                            gelu_enable: bool = False,
                            silu_enable: bool = False,
                            softmax_enable: bool = False) -> Callable[[torch.Tensor], torch.Tensor]:
            """
            Create a reusable BF16 matrix-matrix multiplication with K-partition optimization.
    
            Computes: Y = X @ W + bias [+ GELU/SILU/Softmax]
    
            Where:
                X: Input activation matrix (M x K) in bf16 - provided per call
                W: Weight matrix (N x K) - stored during setup
                Y: Output matrix (M x N) in bf16
                bias: Optional bias vector (N,) in bf16 - stored during setup
    
            Dimension Convention:
            =====================
                M = batch/sequence dimension (number of input vectors)
                K = input features (shared dimension, must be multiple of 64)
                N = output features (must be multiple of 64)
    
            Partitioning Strategy (K-partition optimization):
            =================================================
            Uses dynamic URAM splitting based on bf16_matmat_activation_k_partition:
    
            1. max_N_chunk = min(N, (URAM_NEAR_FULL // K) // 64 * 64)
               - Maximum weight rows that fit in URAM_B
    
            2. Dynamic L calculation for optimal input/output balance:
               - ratio = (max_N_chunk / K) + 1.0
               - L = URAM_FULL / ratio
               - max_M_chunk = min(M, L // K, (URAM_FULL - L) // max_N_chunk)
    
            URAM Memory Map:
            ================
            URAM_A:                              URAM_B:
            ┌─────────────────────────┐          ┌─────────────────────────┐
            │ 0x000                   │          │ 0x000                   │
            │ A_chunk (max_M_chunk×K) │ ◄─Input  │ W_chunk (N_chunk x K)   │ ◄─Weight (stored)
            │ (first L elements)      │          │ (loaded per N iteration)│
            ├─────────────────────────┤          └─────────────────────────┘
            │ L/64 (dynamic)          │
            │ y (1 x N_chunk)         │ ◄─Output (one row, reused)
            └─────────────────────────┘
    
            Processing Flow:
            ================
            For each M_chunk of input rows:
              1. Load A_chunk (max_M_chunk x K) to URAM_A[0x000]
              For each N_chunk of weight rows:
                2. Load W_chunk (N_chunk x K) to URAM_B (from stored params)
                3. Load bias_chunk to BIAS_BRAM (if bias)
                For each row i in M_chunk:
                  4. BF16_DOT_PRODUCT: A[i, :] @ W_chunk.T → y
                  5. Write row to DRAM at output_addr + i * N + N_chunk_offset
    
            Args:
                weight: Weight matrix (N x K) in bf16 format - stored during setup
                M: Batch/sequence dimension (number of input vectors)
                K: Input features (must be multiple of 64)
                N: Output features (must be multiple of 64)
                input_dram_addr: DRAM address for input matrix (M x K) - provided per call
                output_dram_addr: DRAM address for output matrix (M x N)
                program_dram_addr: DRAM address for instruction stream. If None, auto-allocated.
                params_dram_addr: DRAM address for weight and bias. If None, auto-allocated.
                bias: Optional bias vector (N,) in bf16 format - stored during setup
                gelu_enable: Enable GELU activation after multiply (default: False)
                silu_enable: Enable SILU activation after multiply (default: False)
                softmax_enable: Enable row-wise softmax after multiply (default: False)
    
            Returns:
                handler: Callable that takes input (M, K) and returns output (M, N) in bf16 format
            """
            bytes_per_element = 2
    
            # Verify alignment
            if K % UE_VECTOR_SIZE != 0:
                raise ValueError(f"K={K} must be a multiple of {UE_VECTOR_SIZE}")
            if N % UE_VECTOR_SIZE != 0:
                raise ValueError(f"N={N} must be a multiple of {UE_VECTOR_SIZE}")
    
            # Validate mutually exclusive options
            activation_count = sum([gelu_enable, silu_enable, softmax_enable])
            if activation_count > 1:
                raise ValueError("gelu_enable, silu_enable, and softmax_enable are mutually exclusive")
    
            assert weight.dim() == 2, "Weight must be 2D"
            assert weight.dtype == torch.bfloat16, "Weight must be bfloat16"
            assert weight.shape[0] == N and weight.shape[1] == K, f"Weight shape {weight.shape} must match ({N}, {K})"
    
            if bias is not None:
                assert bias.dim() == 1, "Bias must be 1D"
                assert bias.dtype == torch.bfloat16, "Bias must be bfloat16"
                assert bias.shape[0] == N, f"Bias shape {bias.shape[0]} must match N={N}"
    
            op_suffix = ""
            if gelu_enable:
                op_suffix = " + GELU"
            elif silu_enable:
                op_suffix = " + SILU"
            elif softmax_enable:
                op_suffix = " + Softmax"
            print(f"[bf16_matmat{op_suffix}] M={M}, K={K}, N={N}, bias={bias is not None}")
    
            # Auto-allocate params_dram_addr if not provided
            use_auto_mem_management_for_params = False
            if params_dram_addr is None:
                use_auto_mem_management_for_params = True
                params_dram_addr = self.get_params_dram_addr()
            else:
                print(f"Using provided params DRAM address: {params_dram_addr}")
    
            # ===== Write weight and bias to DRAM =====
            current_params_addr = params_dram_addr
    
            # Write weight matrix (N x K)
            weight_dram_addr = current_params_addr
            weight_bytes = N * K * bytes_per_element
            self.dma_write(DMA_DEVICE_H2C, weight_dram_addr, weight.contiguous(), weight_bytes)
            current_params_addr += weight_bytes
    
            # Write bias to DRAM if provided
            bias_dram_addr = None
            if bias is not None:
                bias_dram_addr = current_params_addr
                bias_bytes = N * bytes_per_element
                self.dma_write(DMA_DEVICE_H2C, bias_dram_addr, bias, bias_bytes)
                current_params_addr += bias_bytes
                print(f"[bf16_matmat] Bias: {N} elements, addr=0x{bias_dram_addr:08x}")
    
            params_size = current_params_addr - params_dram_addr
            if use_auto_mem_management_for_params:
                self.allocate_params_dram(params_size)
    
            # ===== Calculate total bytes =====
            total_input_bytes = M * K * bytes_per_element
    
            # ===== Capture instructions =====
            self.start_capture()
            inst_id = 0
    
            if softmax_enable:
                # Softmax path: use different partitioning to accumulate full row before softmax
                N_ALIGNED = (N // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
    
                max_N_chunk = ((URAM_NEAR_FULL_ELEMENTS // K) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
                max_N_chunk = min(max_N_chunk, N)
                assert max_N_chunk >= 1 and max_N_chunk <= N, f"max_N_chunk={max_N_chunk} must be valid"
    
                max_M_chunk = (((URAM_FULL_ELEMENTS - N_ALIGNED) // K) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
                output_row_uram_addr = URAM_START_ADDR + K * (max_M_chunk // UE_VECTOR_SIZE)
    
                max_M_chunk = min(max_M_chunk, M)
                assert max_M_chunk >= 1 and max_M_chunk <= M, f"max_M_chunk={max_M_chunk} must be valid"
    
                print(f"[bf16_matmat + Softmax] max_M_chunk={max_M_chunk}, max_N_chunk={max_N_chunk}")
    
                remaining_M_vectors = M
                input_chunk_dram_addr = input_dram_addr
                output_chunk_dram_addr = output_dram_addr
    
                while remaining_M_vectors > 0:
                    current_max_M_chunk = min(max_M_chunk, remaining_M_vectors)
                    current_M_chunk_bytes = current_max_M_chunk * K * bytes_per_element
    
                    input_uram_addr = URAM_START_ADDR
                    self.ue_memcpy_from_dram(input_chunk_dram_addr, current_M_chunk_bytes, 0, input_uram_addr, URAM_SECTION.URAM_A.value, inst_id)
                    self.wait_queue()
                    inst_id += 1
    
                    for i in range(current_max_M_chunk):
                        remaining_N_chunks = N
                        weight_chunk_offset = weight_dram_addr
                        output_uram_addr = output_row_uram_addr
    
                        clear_max_en = 1
                        while remaining_N_chunks > 0:
                            current_N_chunk = min(max_N_chunk, remaining_N_chunks)
                            current_N_chunk_bytes = current_N_chunk * K * bytes_per_element
    
                            self.ue_memcpy_from_dram(weight_chunk_offset, current_N_chunk_bytes, 0, URAM_START_ADDR, URAM_SECTION.URAM_B.value, inst_id)
                            self.wait_queue()
                            inst_id += 1
    
                            if bias is not None:
                                bias_offset = bias_dram_addr + (N - remaining_N_chunks) * bytes_per_element
                                self.ue_memcpy_from_dram(bias_offset, current_N_chunk * bytes_per_element, MEMCPY_TYPE.BIAS_BRAM.value, 0, 0, inst_id)
                                self.wait_queue()
                                inst_id += 1
    
                            remaining_N_chunks -= current_N_chunk
                            weight_chunk_offset += current_N_chunk_bytes
    
                            self.start_queue(
                                0,  # broadcast_mode
                                clear_max_en,  # clear_max_en
                                1,  # stride_z
                                LALU_MODE.BYPASS.value,  # lalu_mode
                                0,  # lalu_scalar
                                0,  # uram_bram
                                URAM_SECTION.URAM_A.value,  # uram_section
                                0,  # uram_dst_addr
                                0,  # dram_to_uram_cpy_start
                                output_uram_addr,  # uram_wb_addr
                                URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                                UE_MODE.BF16_DOT_PRODUCT,  # mode
                                0,  # data_type
                                input_uram_addr + i * (K // UE_VECTOR_SIZE),  # uram_a_start_addr
                                URAM_START_ADDR,  # uram_b_start_addr
                                K // UE_VECTOR_SIZE,  # uram_length
                                0,  # dma_start_addr
                                current_N_chunk * K,  # dma_length
                                current_N_chunk,  # output_size
                                inst_id,
                                1 if bias is not None else 0
                            )
                            inst_id += 1
                            self.wait_queue()
    
                            clear_max_en = 0
                            output_uram_addr += current_N_chunk // UE_VECTOR_SIZE
    
                        # Softmax: EXP with FMAX_NEGATE
                        self.start_queue(
                            BROADCAST_MODE.FMAX_NEGATE.value,  # broadcast_mode
                            0,  # clear_max_en
                            1,  # stride_z
                            LALU_MODE.MODE_RECIP.value,  # lalu_mode
                            0x9FC00,  # lalu_scalar
                            0,  # uram_bram
                            URAM_SECTION.URAM_A.value,  # uram_section
                            0,  # uram_dst_addr
                            0,  # dram_to_uram_cpy_start
                            output_row_uram_addr,  # uram_wb_addr
                            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                            UE_MODE.EXP,  # mode
                            0,  # data_type
                            output_row_uram_addr,  # uram_a_start_addr
                            0,  # uram_b_start_addr
                            N // UE_VECTOR_SIZE,  # uram_length
                            0,  # dma_start_addr
                            0,  # dma_length
                            0,  # output_size
                            inst_id
                        )
                        inst_id += 1
                        self.wait_queue()
    
                        # Softmax: MUL_BROADCAST
                        self.start_queue(
                            BROADCAST_MODE.LALU_RESULT.value,  # broadcast_mode
                            0,  # clear_max_en
                            1,  # stride_z
                            LALU_MODE.BYPASS.value,  # lalu_mode
                            0,  # lalu_scalar
                            0,  # uram_bram
                            URAM_SECTION.URAM_A.value,  # uram_section
                            0,  # uram_dst_addr
                            0,  # dram_to_uram_cpy_start
                            output_row_uram_addr,  # uram_wb_addr
                            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                            UE_MODE.MUL_BROADCAST,  # mode
                            0,  # data_type
                            output_row_uram_addr,  # uram_a_start_addr
                            0,  # uram_b_start_addr
                            N // UE_VECTOR_SIZE,  # uram_length
                            0,  # dma_start_addr
                            0,  # dma_length
                            0,  # output_size
                            inst_id
                        )
                        inst_id += 1
                        self.wait_queue()
    
                        self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, output_row_uram_addr, output_chunk_dram_addr + i * N * bytes_per_element, N * bytes_per_element, inst_id)
                        self.wait_queue()
                        inst_id += 1
    
                    remaining_M_vectors -= current_max_M_chunk
                    output_chunk_dram_addr += current_max_M_chunk * N * bytes_per_element
                    input_chunk_dram_addr += current_M_chunk_bytes
            else:
                # Non-softmax path: K-partition optimization
                max_N_chunk = ((URAM_NEAR_FULL_ELEMENTS // K) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
                max_N_chunk = min(max_N_chunk, N)
                assert max_N_chunk >= 1 and max_N_chunk <= N, f"max_N_chunk={max_N_chunk} must be valid"
    
                ratio = (max_N_chunk / K) + 1.0
                L = int(URAM_FULL_ELEMENTS / ratio)
                max_M_chunk = min(M, L // K, (URAM_FULL_ELEMENTS - L) // max_N_chunk)
                max_M_chunk = max(1, max_M_chunk)
    
                output_row_uram_addr = L // UE_VECTOR_SIZE
    
                print(f"[bf16_matmat] K-partition: L={L}, max_M_chunk={max_M_chunk}, max_N_chunk={max_N_chunk}")
    
                # Determine LALU mode
                if gelu_enable:
                    dp_lalu_mode = LALU_MODE.GELU.value
                    dp_lalu_scalar = 0
                elif silu_enable:
                    dp_lalu_mode = LALU_MODE.SILU.value
                    dp_lalu_scalar = 0
                else:
                    dp_lalu_mode = LALU_MODE.BYPASS.value
                    dp_lalu_scalar = 0
    
                remaining_M_vectors = M
                input_chunk_dram_addr = input_dram_addr
                output_chunk_dram_addr = output_dram_addr
    
                while remaining_M_vectors > 0:
                    current_max_M_chunk = min(max_M_chunk, remaining_M_vectors)
                    current_M_chunk_bytes = current_max_M_chunk * K * bytes_per_element
    
                    input_uram_addr = URAM_START_ADDR
                    self.ue_memcpy_from_dram(input_chunk_dram_addr, current_M_chunk_bytes, 0, input_uram_addr, URAM_SECTION.URAM_A.value, inst_id)
                    self.wait_queue()
                    inst_id += 1
    
                    remaining_N_chunks = N
                    weight_chunk_offset = weight_dram_addr
    
                    output_chunk_dram_offset_in_chunk_row = output_chunk_dram_addr
                    while remaining_N_chunks > 0:
                        current_N_chunk = min(max_N_chunk, remaining_N_chunks)
                        current_N_chunk_bytes = current_N_chunk * K * bytes_per_element
    
                        self.ue_memcpy_from_dram(weight_chunk_offset, current_N_chunk_bytes, 0, URAM_START_ADDR, URAM_SECTION.URAM_B.value, inst_id)
                        self.wait_queue()
                        inst_id += 1
    
                        if bias is not None:
                            bias_offset = bias_dram_addr + (N - remaining_N_chunks) * bytes_per_element
                            self.ue_memcpy_from_dram(bias_offset, current_N_chunk * bytes_per_element, MEMCPY_TYPE.BIAS_BRAM.value, 0, 0, inst_id)
                            self.wait_queue()
                            inst_id += 1
    
                        for i in range(current_max_M_chunk):
                            self.start_queue(
                                0,  # broadcast_mode
                                0,  # clear_max_en
                                1,  # stride_z
                                dp_lalu_mode,  # lalu_mode
                                dp_lalu_scalar,  # lalu_scalar
                                0,  # uram_bram
                                URAM_SECTION.URAM_A.value,  # uram_section
                                0,  # uram_dst_addr
                                0,  # dram_to_uram_cpy_start
                                output_row_uram_addr,  # uram_wb_addr
                                URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                                UE_MODE.BF16_DOT_PRODUCT,  # mode
                                0,  # data_type
                                input_uram_addr + i * (K // UE_VECTOR_SIZE),  # uram_a_start_addr
                                URAM_START_ADDR,  # uram_b_start_addr
                                K // UE_VECTOR_SIZE,  # uram_length
                                0,  # dma_start_addr
                                current_N_chunk * K,  # dma_length
                                current_N_chunk,  # output_size
                                inst_id,
                                1 if bias is not None else 0
                            )
                            inst_id += 1
                            self.wait_queue()
    
                            self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, output_row_uram_addr, output_chunk_dram_offset_in_chunk_row + i * N * bytes_per_element, current_N_chunk * bytes_per_element, inst_id)
                            self.wait_queue()
                            inst_id += 1
    
                        remaining_N_chunks -= current_N_chunk
                        output_chunk_dram_offset_in_chunk_row += current_N_chunk * bytes_per_element
                        weight_chunk_offset += current_N_chunk_bytes
    
                    output_chunk_dram_addr += current_max_M_chunk * N * bytes_per_element
                    remaining_M_vectors -= current_max_M_chunk
                    input_chunk_dram_addr += current_M_chunk_bytes
    
            # Finish capture and write instruction stream to DRAM
            self.stop_capture()
            self.generate_instruction_halt()
    
            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
                self.allocate_program_dram(program_size_bytes)
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
                self.write_captured_instructions_to_dram(program_dram_addr)
    
            def handler(input_matrix) -> torch.Tensor:
                """
                Run BF16 matrix-matrix multiply: Y = X @ W + bias [+ GELU/SILU/Softmax]
    
                Supports DeviceTensor for DMA cache optimization.
    
                Args:
                    input_matrix: Input (M, K) - torch.Tensor or DeviceTensor
    
                Returns:
                    Result matrix (M, N) in bf16 format
                """
                # Extract tensor data and check if DMA can be skipped
                if isinstance(input_matrix, DeviceTensor):
                    input_data = input_matrix._data
                    skip_dma = not input_matrix.needs_dma(input_dram_addr)
                else:
                    input_data = input_matrix
                    skip_dma = False
    
                assert input_data.dtype == torch.bfloat16, "Input matrix must be in bf16 format"
    
                if input_data.dim() == 1:
                    input_data = input_data.unsqueeze(0)
    
                input_M, input_K = input_data.shape
                if input_M != M:
                    raise ValueError(f"Input matrix has {input_M} rows, expected M={M}")
                if input_K != K:
                    raise ValueError(f"Input matrix has {input_K} columns, expected K={K}")
    
                # DMA input to DRAM (skip if DeviceTensor is already synced)
                if skip_dma:
                    print(f"[DMA cache hit] Skipping input DMA to {hex(input_dram_addr)}")
                else:
                    if isinstance(input_matrix, DeviceTensor):
                        input_matrix.sync(input_dram_addr)
                    else:
                        self.dma_write(DMA_DEVICE_H2C, input_dram_addr, input_data.flatten(), total_input_bytes)
    
                # Start executing from DRAM and wait for completion
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
    
                self.report_timing_and_instruction_count()
                # Report timing and flop rate
                flops = M * K * N * 2
                if bias is not None:
                    flops += M * N
                if softmax_enable:
                    flops += M * N * 5
                if gelu_enable or silu_enable:
                    flops += M * N * 8
                flop_rate = self.report_flop_rate_gflops(flops)
                print(f"[bf16_matmat{op_suffix}] {self.report_latency_in_us():.3f} us, {flop_rate:.3f} GFLOPS")
    
                # Return DeviceTensor if input was DeviceTensor, else return raw tensor
                output_tensor = DeviceTensor((M, N), ue=self, dram_addr=output_dram_addr)
                if isinstance(input_matrix, DeviceTensor):
                    return output_tensor
                else:
                    return output_tensor.data
    
            return handler
    
        def bf16_matmat_activation(self, M: int, K: int, N: int,
                                       input_dram_addr: int,
                                       weight_dram_addr: int,
                                       output_dram_addr: int,
                                       program_dram_addr: Optional[int] = None,
                                       gelu_enable: bool = False,
                                       silu_enable: bool = False,
                                       softmax_enable: bool = False) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
            """
            Create a reusable BF16 matrix-matrix multiplication operation where weights are activations.
    
            Unlike bf16_matmat, weights are NOT stored during setup. Instead, both input and weight
            matrices are provided per call (both are activations).
    
            Computes: C = A @ B.T, optionally followed by GELU, SILU, or row-wise softmax.
    
            Where:
                A: Input activation matrix (M x K) in bf16 - provided per call
                B: Weight matrix (N x K) in bf16 - provided per call, tranpose is implicitly handled by the hardware
                C: Output matrix (M x N) in bf16
    
            Dimension Convention:
            =====================
                M = batch/sequence dimension (number of input vectors)
                K = input features (shared dimension)
                N = output features
    
            Weight Partitioning:
            ====================
            If N * K exceeds URAM_B capacity, the weight matrix is partitioned by
            output features (N dimension) into chunks that fit.
    
            N_chunk = min(N, URAM_B_CAPACITY // K) aligned to 64
    
            For each weight chunk:
              1. Load B_chunk^T (K x N_chunk) to URAM_B
              2. Process all M input vectors
              3. Write output chunk (M x N_chunk) to DRAM at row offset
    
            URAM_B Partitioning:
            ====================
            Each half of URAM_B: 2048 * 64 = 131072 elements
    
            Partition size P = min(131072 // K, 131072 // N_chunk, M)
    
            Args:
                M: Batch/sequence dimension (number of input vectors)
                K: Input features (must be multiple of 64)
                N: Output features (must be multiple of 64)
                input_dram_addr: DRAM address for input matrix (M x K) - provided per call
                weight_dram_addr: DRAM address for weight matrix (N x K) - provided per call, tranpose is implicitly handled by the hardware
                output_dram_addr: DRAM address for output matrix (M x N)
                program_dram_addr: DRAM address for instruction stream. If None, auto-allocated.
                gelu_enable: Enable GELU activation after multiply (default: False)
                silu_enable: Enable SILU activation after multiply (default: False)
                softmax_enable: Enable row-wise softmax after multiply (default: False)
    
            Returns:
                handler: Callable that takes (input_matrix, weight_matrix) and returns output (M, N) in bf16 format
            """
            bytes_per_element = 2
    
            # Verify alignment
            if K % UE_VECTOR_SIZE != 0:
                raise ValueError(f"K={K} must be a multiple of {UE_VECTOR_SIZE}")
            if N % UE_VECTOR_SIZE != 0:
                raise ValueError(f"N={N} must be a multiple of {UE_VECTOR_SIZE}")
    
            # Validate mutually exclusive options: when softmax is enabled, gelu and silu cannot be enabled
            if softmax_enable:
                if gelu_enable or silu_enable:
                    raise ValueError("When softmax_enable=True, gelu_enable and silu_enable must be False")
            activation_count = sum([gelu_enable, silu_enable, softmax_enable])
            if activation_count > 1:
                raise ValueError("gelu_enable, silu_enable, and softmax_enable are mutually exclusive")
    
            op_suffix = ""
            if gelu_enable:
                op_suffix = " + GELU"
            elif silu_enable:
                op_suffix = " + SILU"
            elif softmax_enable:
                op_suffix = " + Softmax"
            print(f"[bf16_matmat_activation{op_suffix}] M={M}, K={K}, N={N}")
    
            # ===== Calculate URAM_B partitioning =====
            # Maximum N_chunk that fits in URAM_B with K columns
            max_N_chunk = ((URAM_NEAR_FULL_ELEMENTS // K) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE # Lower to nearest multiple of 64
            max_N_chunk = min(max_N_chunk, N) # matrix is loaded into URAM_B with chunks of K * max_N_chunk elements
            assert max_N_chunk >= 1 and max_N_chunk <= N, f"max_N_chunk={max_N_chunk} must be greater than 0 and less than N={N}"
    
            max_M_chunk = min(M, URAM_HALF_ELEMENTS // K, URAM_HALF_ELEMENTS // N) # Assume that matrix is loaded into URAM_B fully, we'll do this with multiple iterations
            assert max_M_chunk >= 1 and max_M_chunk <= M, f"max_M_chunk={max_M_chunk} must be greater than 0 and less than M={M}"
    
            print(f"URAMB can have: {max_N_chunk}x{K} of {N}x{K}")
            if max_N_chunk < N:
                print("softmax needs to be handled properly in this case!!!")
                print("for every max_M_chunk need to complete KxN operations")
    
            # Uram A first half is max_M_chunk x K and second half is K x max_N_chunk
            # Max calculated matrices by hw is thus (max_M_chunk x K) and (K x max_N_chunk)
            # iterations are 0:max_M_chunk:M and 0:max_N_chunk:N
    
            # ===== Calculate total bytes =====
            total_input_bytes = M * K * bytes_per_element
            total_weight_bytes = K * N * bytes_per_element
            total_output_bytes = M * N * bytes_per_element
    
            # Determine LALU mode for dot product (GELU/SILU only if no softmax)
            if gelu_enable:
                dp_lalu_mode = LALU_MODE.GELU.value
                dp_lalu_scalar = 0
            elif silu_enable:
                dp_lalu_mode = LALU_MODE.SILU.value
                dp_lalu_scalar = 0
            else:
                dp_lalu_mode = LALU_MODE.BYPASS.value
                dp_lalu_scalar = 0
    
            # ===== Capture instructions =====
            self.start_capture()
            inst_id = 0
    
            # Process each weight chunk
            remaining_M_vectors = M
            input_chunk_dram_addr = input_dram_addr
            output_chunk_dram_addr = output_dram_addr
            while remaining_M_vectors > 0:
    
                current_max_M_chunk = min(max_M_chunk, remaining_M_vectors)
                current_M_chunk_bytes = current_max_M_chunk * K * bytes_per_element
    
                input_uram_addr = URAM_START_ADDR
                output_uram_addr = URAM_HALFWAY_ADDR
                self.ue_memcpy_from_dram(input_chunk_dram_addr, current_M_chunk_bytes, 0, input_uram_addr, URAM_SECTION.URAM_A.value, inst_id)
                self.wait_queue()
                inst_id += 1
    
                # For every output row
                for i in range(current_max_M_chunk):
    
                    remaining_N_chunks = N
                    weight_chunk_dram_addr = weight_dram_addr
                    clear_max_en = 1
    
                    one_by_N_uram_addr = output_uram_addr
                    # weight matrix chunking is done here
                    while remaining_N_chunks > 0:
                        current_N_chunk = min(max_N_chunk, remaining_N_chunks)
                        current_N_chunk_bytes = current_N_chunk * K * bytes_per_element
    
                        # weight is always copied to URAM_B from start of URAM_B
                        self.ue_memcpy_from_dram(weight_chunk_dram_addr, current_N_chunk_bytes, 0, URAM_START_ADDR, URAM_SECTION.URAM_B.value, inst_id)
                        self.wait_queue()
                        inst_id += 1
    
                        # BF16 dot product
                        self.start_queue(
                            0,  # broadcast_mode
                            clear_max_en,  # clear_max_en
                            1,  # stride_z
                            dp_lalu_mode,  # lalu_mode
                            dp_lalu_scalar,  # lalu_scalar
                            0,  # uram_bram
                            URAM_SECTION.URAM_A.value,  # uram_section
                            0,  # uram_dst_addr
                            0,  # dram_to_uram_cpy_start
                            output_uram_addr,  # uram_wb_addr
                            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                            UE_MODE.BF16_DOT_PRODUCT,  # mode
                            0,  # data_type
                            input_uram_addr,  # uram_a_start_addr
                            URAM_START_ADDR,  # uram_b_start_addr
                            K // UE_VECTOR_SIZE,  # uram_length
                            0,  # dma_start_addr not used
                            current_N_chunk * K,  # dma_length
                            current_N_chunk,  # output_size
                            inst_id
                        )
                        inst_id += 1
                        self.wait_queue()
    
                        clear_max_en = 0 # clear to generate pulse
    
                        output_uram_addr += current_N_chunk // UE_VECTOR_SIZE # move by 1xN_chunk elements
                        remaining_N_chunks -= current_N_chunk
                        weight_chunk_dram_addr += current_N_chunk_bytes
    
                    # fmax has been done already, now do softmax
                    if softmax_enable:
                        self.start_queue(
                            BROADCAST_MODE.FMAX_NEGATE.value,  # broadcast_mode
                            0,  # clear_max not used
                            1,  # stride_z
                            LALU_MODE.MODE_RECIP.value,  # lalu_mode
                            0x9FC00,  # lalu_scalar (1.0 in bf19 format)
                            0,  # uram_bram
                            URAM_SECTION.URAM_A.value,  # uram_section
                            0,  # uram_dst_addr
                            0,  # dram_to_uram_cpy_start
                            one_by_N_uram_addr,  # uram_wb_addr
                            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                            UE_MODE.EXP,  # mode
                            0,  # data_type
                            one_by_N_uram_addr,  # uram_a_start_addr
                            0,  # uram_b_start_addr (not used)
                            N // UE_VECTOR_SIZE,  # uram_length
                            0,  # dma_start_addr (not used)
                            0,  # dma_length (not used)
                            0,  # output_size (not used)
                            inst_id
                        )
                        inst_id += 1
                        self.wait_queue()
    
                        # Pass 2: MUL_BROADCAST mode
                        self.start_queue(
                            BROADCAST_MODE.LALU_RESULT.value,  # broadcast_mode
                            0,  # clear_max_en
                            1,  # stride_z
                            LALU_MODE.BYPASS.value,  # lalu_mode
                            0,  # lalu_scalar
                            0,  # uram_bram
                            URAM_SECTION.URAM_A.value,  # uram_section
                            0,  # uram_dst_addr
                            0,  # dram_to_uram_cpy_start
                            one_by_N_uram_addr,  # uram_wb_addr
                            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                            UE_MODE.MUL_BROADCAST,  # mode
                            0,  # data_type
                            one_by_N_uram_addr,  # uram_a_start_addr
                            0,  # uram_b_start_addr
                            N // UE_VECTOR_SIZE,  # uram_length
                            0,  # dma_start_addr (not used)
                            0,  # dma_length (not used)
                            0,  # output_size (not used)
                            inst_id
                        )
                        inst_id += 1
                        self.wait_queue()
    
                    input_uram_addr += K // UE_VECTOR_SIZE # move by 1xK elements
    
                # current_max_M_chunk x N elements are in URAM_HALFWAY_ADDR, copy to DRAM
                self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, URAM_HALFWAY_ADDR, output_chunk_dram_addr, current_max_M_chunk * N * bytes_per_element, inst_id)
                self.wait_queue()
                inst_id += 1
    
                output_chunk_dram_addr += current_max_M_chunk * N * bytes_per_element
                input_chunk_dram_addr += current_M_chunk_bytes
                remaining_M_vectors -= current_max_M_chunk
    
            # Finish capture and write instruction stream to DRAM
            self.stop_capture()
            self.generate_instruction_halt()
    
            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
                self.allocate_program_dram(program_size_bytes)
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
                self.write_captured_instructions_to_dram(program_dram_addr)
    
            def handler(input_matrix, weight_matrix):
                """
                Run BF16 matrix-matrix multiply: Y = X @ W.T [+ GELU/Softmax]
                Both input and weight matrices are provided per call (both are activations).
    
                Supports DeviceTensor for DMA cache optimization.
    
                Args:
                    input_matrix: Input matrix (M, K) - torch.Tensor or DeviceTensor
                    weight_matrix: Weight matrix (N, K) - torch.Tensor or DeviceTensor
    
                Returns:
                    DeviceTensor if any input was DeviceTensor, else raw tensor (M, N)
                """
                # Extract tensor data and check if DMA can be skipped
                is_input_device = isinstance(input_matrix, DeviceTensor)
                is_weight_device = isinstance(weight_matrix, DeviceTensor)
    
                if is_input_device:
                    input_data = input_matrix._data
                    skip_input = not input_matrix.needs_dma(input_dram_addr)
                else:
                    input_data = input_matrix
                    skip_input = False
    
                if is_weight_device:
                    weight_data = weight_matrix._data
                    skip_weight = not weight_matrix.needs_dma(weight_dram_addr)
                else:
                    weight_data = weight_matrix
                    skip_weight = False
    
                assert input_data.dtype == torch.bfloat16, "Input matrix must be in bf16 format"
                assert weight_data.dtype == torch.bfloat16, "Weight matrix must be in bf16 format"
    
                if input_data.dim() == 1:
                    input_data = input_data.unsqueeze(0)
    
                input_M, input_K = input_data.shape
                if input_M != M:
                    raise ValueError(f"Input matrix has {input_M} rows, expected M={M}")
                if input_K != K:
                    raise ValueError(f"Input matrix has {input_K} columns, expected K={K}")
    
                weight_N, weight_K = weight_data.shape
                if weight_K != K:
                    raise ValueError(f"Weight matrix has {weight_K} rows, expected K={K}")
                if weight_N != N:
                    raise ValueError(f"Weight matrix has {weight_N} columns, expected N={N}")
    
                # DMA input and weight to DRAM (skip if DeviceTensor already synced)
                if skip_input:
                    print(f"[DMA cache hit] Skipping input DMA to {hex(input_dram_addr)}")
                else:
                    if isinstance(input_matrix, DeviceTensor):
                        input_matrix.sync(input_dram_addr)
                    else:
                        self.dma_write(DMA_DEVICE_H2C, input_dram_addr, input_data.flatten(), total_input_bytes)
    
                if skip_weight:
                    print(f"[DMA cache hit] Skipping weight DMA to {hex(weight_dram_addr)}")
                else:
                    if isinstance(weight_matrix, DeviceTensor):
                        weight_matrix.sync(weight_dram_addr)
                    else:
                        self.dma_write(DMA_DEVICE_H2C, weight_dram_addr, weight_data.contiguous().flatten(), total_weight_bytes)
    
                # Start executing from DRAM and wait for completion
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
                self.report_timing_and_instruction_count()
                # Matrix-matrix multiply: 2*M*K*N FLOPs + softmax ops (22 FLOPs per element)
                total_flops = (2 * M * K * N + 22 * M * N) if softmax_enable else 2 * M * K * N
                print(f"[bf16_matmat_activation] {self.report_latency_in_us():.3f} us, {self.report_flop_rate_gflops(total_flops):.3f} GFLOPS")
    
                # Return DeviceTensor if any input was DeviceTensor, else return raw tensor
                output_tensor = DeviceTensor((M, N), ue=self, dram_addr=output_dram_addr)
                if is_input_device or is_weight_device:
                    return output_tensor
                else:
                    return output_tensor.data
    
            return handler
    
        def batched_matmat_mul(self, M: int, K: int, N: int,
                                        input_dram_addr_batch: int,
                                        weight_dram_addr_batch: int,
                                        output_dram_addr_batch: int,
                                        batch_size: int = 1,
                                        program_dram_addr: Optional[int] = None,
                                        bias_enable: bool = False,
                                        bias_matrix_enable: bool = False,
                                        bias_dram_addr_batch: Optional[int] = None,
                                        softmax_enable: bool = False,
                                        gelu_enable: bool = False,
                                        silu_enable: bool = False) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
            """
            Create a reusable Batched BF16 matrix-matrix multiplication.
            
            Computes: C = A @ B.T [+ GELU/SILU/Softmax] for a batch of inputs.
            
            Args:
                M, K, N: Matrix dimensions (M=Rows, K=Inner, N=Cols)
                input_dram_addr_batch: Base DRAM address for Input Batch
                weight_dram_addr_batch: Base DRAM address for Weight Batch
                output_dram_addr_batch: Base DRAM address for Output Batch
                batch_size: Number of matrix multiplications to perform
            """
            bytes_per_element = 2
    
            # Verify alignment
            if K % UE_VECTOR_SIZE != 0:
                raise ValueError(f"K={K} must be a multiple of {UE_VECTOR_SIZE}")
            if N % UE_VECTOR_SIZE != 0:
                raise ValueError(f"N={N} must be a multiple of {UE_VECTOR_SIZE}")
    
            if bias_enable is True:
                assert bias_dram_addr_batch is not None, "bias_dram_addr_batch must be provided when bias_enable is True"
                if bias_matrix_enable is True:
                    total_bias_bytes = M * N * bytes_per_element
                else:
                    total_bias_bytes = N * bytes_per_element
    
            # Validate mutually exclusive options
            activation_count = sum([gelu_enable, silu_enable, softmax_enable])
            if activation_count > 1:
                raise ValueError("gelu_enable, silu_enable, and softmax_enable are mutually exclusive")
    
            op_suffix = ""
            if gelu_enable: op_suffix = " + GELU"
            elif silu_enable: op_suffix = " + SILU"
            elif softmax_enable: op_suffix = " + Softmax"
    
            print(f"[batched_matmat_mul{op_suffix}] Batch={batch_size}, M={M}, K={K}, N={N}")
    
            # ===== Calculate total bytes per batch item =====
            input_batch_stride = M * K * bytes_per_element
            weight_batch_stride = K * N * bytes_per_element
            output_batch_stride = M * N * bytes_per_element
            bias_batch_stride = (M * N if bias_matrix_enable else N) * bytes_per_element
    
            # ===== Get program DRAM address BEFORE capture (needed for absolute jump addresses) =====
            auto_allocated = False
            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                auto_allocated = True
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
    
            # ===== Capture instructions =====
            self.start_capture()
            inst_id = 0
    
            # ------------------------------------------------------------------
            # ISA-based outer loop over batch using general-purpose registers
            # ------------------------------------------------------------------
            # General-purpose ISA registers are 32 bits wide: regs 0..15.
            # We start allocating from 1 (reg 0 is a hard-wired zero register).
            general_reg_src = 1
    
            def _alloc_isa_reg() -> int:
                nonlocal general_reg_src
                if general_reg_src > 15:
                    raise ValueError("Exceeded available ISA registers (max 15)")
                reg_idx = general_reg_src
                general_reg_src += 1
                return reg_idx
    
            # Set general-purpose register for input_dram_addr
            input_dram_addr_reg = _alloc_isa_reg()
            self.generate_instruction_add_set(input_dram_addr_reg, input_dram_addr_batch)
    
            # Set general-purpose register for weight_dram_addr
            weight_dram_addr_reg = _alloc_isa_reg()
            self.generate_instruction_add_set(weight_dram_addr_reg, weight_dram_addr_batch)
    
            # Set general-purpose register for output_dram_addr
            output_dram_addr_reg = _alloc_isa_reg()
            self.generate_instruction_add_set(output_dram_addr_reg, output_dram_addr_batch)
    
            # Optional bias_dram_addr register
            bias_dram_addr_reg = None
            if bias_enable:
                bias_dram_addr_reg = _alloc_isa_reg()
                self.generate_instruction_add_set(bias_dram_addr_reg, bias_dram_addr_batch)
    
            # Loop register and loop count
            loop_reg = _alloc_isa_reg()
            self.generate_instruction_add_set(loop_reg, batch_size)
    
            # Set general-purpose register for temp_dram_addr
            temp_dram_addr_reg = _alloc_isa_reg()
            input_chunk_dram_addr_reg = _alloc_isa_reg()
            output_chunk_dram_addr_reg = _alloc_isa_reg()
            weight_chunk_dram_addr_reg = _alloc_isa_reg()
    
            # Record instruction index at the beginning of the loop body.
            # JUMP will target this instruction index.
            loop_start_inst_index = self.capture_count
    
            # Counters for loop statistics (initialized for softmax path)
            outer_loop_count = 0  # x: number of times "while remaining_M_vectors > 0:" runs
            for_loop_total = 0     # y: total number of "for i in range(current_max_M_chunk):" iterations
            inner_loop_total = 0   # z: total number of "while remaining_N_chunks > 0:" iterations
    
            # Iterate through the batch using ISA loop
            input_dram_addr = input_dram_addr_batch
            weight_dram_addr = weight_dram_addr_batch
            output_dram_addr = output_dram_addr_batch
            
            if bias_enable:
                bias_dram_addr = bias_dram_addr_batch
    
            if softmax_enable:
                # print("softmax needs to be handled properly in this case!!!")
                N_ALIGNED = (N // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
    
                max_N_chunk = ((URAM_NEAR_FULL_ELEMENTS // K) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE 
                max_N_chunk = min(max_N_chunk, N)
                
                max_M_chunk = (((URAM_FULL_ELEMENTS - N_ALIGNED) // K) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
                output_row_uram_addr = URAM_START_ADDR + K * (max_M_chunk // UE_VECTOR_SIZE)
    
                max_M_chunk = min(max_M_chunk, M)
                remaining_M_vectors = M
    
                # input_chunk_dram_addr = input_dram_addr
                # output_chunk_dram_addr = output_dram_addr
                self.generate_instruction_add_reg(input_chunk_dram_addr_reg, input_dram_addr_reg, 0)
                self.generate_instruction_add_reg(output_chunk_dram_addr_reg, output_dram_addr_reg, 0)
    
                while remaining_M_vectors > 0:
                    outer_loop_count += 1
                    current_max_M_chunk = min(max_M_chunk, remaining_M_vectors)
                    current_M_chunk_bytes = current_max_M_chunk * K * bytes_per_element
    
                    input_uram_addr = URAM_START_ADDR
                    self.ue_memcpy_from_dram(0, current_M_chunk_bytes, 0, input_uram_addr, URAM_SECTION.URAM_A.value, inst_id, general_reg_src=input_chunk_dram_addr_reg)
                    self.wait_queue()
                    inst_id += 1
    
                    for i in range(current_max_M_chunk):
                        for_loop_total += 1
                        remaining_N_chunks = N
                        # weight_chunk_dram_addr = weight_dram_addr
                        self.generate_instruction_add_reg(weight_chunk_dram_addr_reg, weight_dram_addr_reg, 0)
                        output_uram_addr = output_row_uram_addr
                        clear_max_en = 1
                        
                        while remaining_N_chunks > 0:
                            inner_loop_total += 1
                            current_N_chunk = min(max_N_chunk, remaining_N_chunks)
                            current_N_chunk_bytes = current_N_chunk * K * bytes_per_element
    
                            self.ue_memcpy_from_dram(0, current_N_chunk_bytes, 0, URAM_START_ADDR, URAM_SECTION.URAM_B.value, inst_id, general_reg_src=weight_chunk_dram_addr_reg)
                            self.wait_queue()
                            inst_id += 1
    
                            if bias_enable is True:
                                if bias_matrix_enable is True:
                                    # bias_offset = bias_dram_addr + ((N - remaining_N_chunks) + N * (M - remaining_M_vectors + i)) * bytes_per_element
                                    self.generate_instruction_add_imm(bias_dram_addr_reg, ((N - remaining_N_chunks) + N * (M - remaining_M_vectors + i)) * bytes_per_element, temp_dram_addr_reg)
                                else:
                                    # bias_offset = bias_dram_addr + (N - remaining_N_chunks) * bytes_per_element
                                    self.generate_instruction_add_imm(bias_dram_addr_reg, (N - remaining_N_chunks) * bytes_per_element, temp_dram_addr_reg)
                                self.ue_memcpy_from_dram(0, current_N_chunk * bytes_per_element, MEMCPY_TYPE.BIAS_BRAM.value, 0, 0, inst_id, general_reg_src=temp_dram_addr_reg)
                                self.wait_queue()
                                inst_id += 1
    
                            remaining_N_chunks -= current_N_chunk
                            # weight_chunk_dram_addr += current_N_chunk_bytes
                            self.generate_instruction_add_imm(weight_chunk_dram_addr_reg, current_N_chunk_bytes)
    
                            self.start_queue(
                                0, clear_max_en, 1, LALU_MODE.BYPASS.value, 0, 0,
                                URAM_SECTION.URAM_A.value, 0, 0, output_uram_addr,
                                URAM_WRITE_SRC.URAM_WRITE_BACK.value, UE_MODE.BF16_DOT_PRODUCT, 0,
                                input_uram_addr + i * (K // UE_VECTOR_SIZE), URAM_START_ADDR,
                                K // UE_VECTOR_SIZE, 0, current_N_chunk * K, current_N_chunk,
                                inst_id, 1 if bias_enable else 0
                            )
                            inst_id += 1
                            self.wait_queue()
                            clear_max_en = 0
                            output_uram_addr += current_N_chunk // UE_VECTOR_SIZE
    
                        # FMAX / SOFTMAX Passes
                        self.start_queue(
                            BROADCAST_MODE.FMAX_NEGATE.value, 0, 1, LALU_MODE.MODE_RECIP.value, 0x9FC00, 0,
                            URAM_SECTION.URAM_A.value, 0, 0, output_row_uram_addr,
                            URAM_WRITE_SRC.URAM_WRITE_BACK.value, UE_MODE.EXP, 0,
                            output_row_uram_addr, 0, N // UE_VECTOR_SIZE, 0, 0, 0, inst_id
                        )
                        inst_id += 1
                        self.wait_queue()
    
                        self.start_queue(
                            BROADCAST_MODE.LALU_RESULT.value, 0, 1, LALU_MODE.BYPASS.value, 0, 0,
                            URAM_SECTION.URAM_A.value, 0, 0, output_row_uram_addr,
                            URAM_WRITE_SRC.URAM_WRITE_BACK.value, UE_MODE.MUL_BROADCAST, 0,
                            output_row_uram_addr, 0, N // UE_VECTOR_SIZE, 0, 0, 0, inst_id
                        )
                        inst_id += 1
                        self.wait_queue()
    
                        # Keep ISA registers in sync with output_chunk_dram_addr + i * N * bytes_per_element
                        self.generate_instruction_add_imm(output_chunk_dram_addr_reg, i * N * bytes_per_element, temp_dram_addr_reg)
    
                        self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, output_row_uram_addr, 0, N * bytes_per_element, inst_id, general_reg_src=temp_dram_addr_reg)
                        self.wait_queue()
                        inst_id += 1
    
                    remaining_M_vectors -= current_max_M_chunk
                    # Keep ISA registers in sync with output_chunk_dram_addr += current_max_M_chunk * N * bytes_per_element
                    self.generate_instruction_add_imm(output_chunk_dram_addr_reg, current_max_M_chunk * N * bytes_per_element)
                    # Keep ISA registers in sync with input_chunk_dram_addr += current_M_chunk_bytes
                    self.generate_instruction_add_imm(input_chunk_dram_addr_reg, current_M_chunk_bytes)
    
            else:
                # ===== Standard Path (No Softmax) =====
                max_N_chunk = ((URAM_NEAR_FULL_ELEMENTS // K) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
                max_N_chunk = min(max_N_chunk, N)
                
                # Optimization logic for M_chunk
                ratio = (max_N_chunk / K) + 1.0
                L = int(URAM_FULL_ELEMENTS / ratio)
                max_M_chunk = min(M, L // K, (URAM_FULL_ELEMENTS - L) // max_N_chunk)
    
                # Determine LALU mode
                if gelu_enable:
                    dp_lalu_mode = LALU_MODE.GELU.value
                elif silu_enable:
                    dp_lalu_mode = LALU_MODE.SILU.value
                else:
                    dp_lalu_mode = LALU_MODE.BYPASS.value
                dp_lalu_scalar = 0
    
                remaining_M_vectors = M
                input_chunk_dram_addr = input_dram_addr
                output_chunk_dram_addr = output_dram_addr
                
                while remaining_M_vectors > 0:
                    current_max_M_chunk = min(max_M_chunk, remaining_M_vectors)
                    current_M_chunk_bytes = current_max_M_chunk * K * bytes_per_element
    
                    input_uram_addr = URAM_START_ADDR
                    output_uram_addr = L // UE_VECTOR_SIZE
                    
                    # Load Input Chunk
                    self.ue_memcpy_from_dram(input_chunk_dram_addr, current_M_chunk_bytes, 0, input_uram_addr, URAM_SECTION.URAM_A.value, inst_id)
                    self.wait_queue()
                    inst_id += 1
    
                    remaining_N_chunks = N
                    weight_chunk_dram_addr = weight_dram_addr
                    output_chunk_dram_offset_in_chunk_row = output_chunk_dram_addr
                    
                    while remaining_N_chunks > 0:
                        current_N_chunk = min(max_N_chunk, remaining_N_chunks)
                        current_N_chunk_bytes = current_N_chunk * K * bytes_per_element
    
                        # Load Weight Chunk
                        self.ue_memcpy_from_dram(weight_chunk_dram_addr, current_N_chunk_bytes, 0, URAM_START_ADDR, URAM_SECTION.URAM_B.value, inst_id)
                        self.wait_queue()
                        inst_id += 1
    
                        if bias_enable is True:
                            if bias_matrix_enable is False:
                                bias_offset = bias_dram_addr + (N - remaining_N_chunks) * bytes_per_element
                                self.ue_memcpy_from_dram(bias_offset, current_N_chunk * bytes_per_element, MEMCPY_TYPE.BIAS_BRAM.value, 0, 0, inst_id)
                                self.wait_queue()
                                inst_id += 1
    
                        for i in range(current_max_M_chunk):
                            if bias_enable is True and bias_matrix_enable is True:
                                bias_offset = bias_dram_addr + ((N - remaining_N_chunks) + N * (M - remaining_M_vectors + i)) * bytes_per_element
                                self.ue_memcpy_from_dram(bias_offset, current_N_chunk * bytes_per_element, MEMCPY_TYPE.BIAS_BRAM.value, 0, 0, inst_id)
                                self.wait_queue()
                                inst_id += 1
    
                            self.start_queue(
                                0, 0, 1, dp_lalu_mode, dp_lalu_scalar, 0,
                                URAM_SECTION.URAM_A.value, 0, 0, output_uram_addr,
                                URAM_WRITE_SRC.URAM_WRITE_BACK.value, UE_MODE.BF16_DOT_PRODUCT, 0,
                                input_uram_addr + i * (K // UE_VECTOR_SIZE), URAM_START_ADDR,
                                K // UE_VECTOR_SIZE, 0, current_N_chunk * K, current_N_chunk,
                                inst_id, 1 if bias_enable else 0
                            )
                            inst_id += 1
                            self.wait_queue()
    
                            self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, output_uram_addr, output_chunk_dram_offset_in_chunk_row + i * N * bytes_per_element, current_N_chunk * bytes_per_element, inst_id)
                            self.wait_queue()
                            inst_id += 1
    
                        remaining_N_chunks -= current_N_chunk
                        output_chunk_dram_offset_in_chunk_row += current_N_chunk * bytes_per_element
                        weight_chunk_dram_addr += current_N_chunk_bytes
    
                    output_chunk_dram_addr += current_max_M_chunk * N * bytes_per_element
                    remaining_M_vectors -= current_max_M_chunk
                    input_chunk_dram_addr += current_M_chunk_bytes
    
            self.generate_instruction_add_imm(input_dram_addr_reg, input_batch_stride)
            self.generate_instruction_add_imm(weight_dram_addr_reg, weight_batch_stride)
            self.generate_instruction_add_imm(output_dram_addr_reg, output_batch_stride)
            if bias_enable:
                self.generate_instruction_add_imm(bias_dram_addr_reg, bias_batch_stride)
    
            # Decrement loop counter
            self.generate_instruction_add_dec(loop_reg)
            # Compute physical DRAM address of loop body
            loop_body_dram_addr = ( program_dram_addr + loop_start_inst_index * 32)
            print(f"Loop body instruction index: {loop_start_inst_index} DRAM address: {loop_body_dram_addr}")
            # Jump back to the start of the loop body while loop_reg != 0
            self.generate_instruction_jump(loop_body_dram_addr, JUMP_MODE_JNZ, loop_reg)    
    
            # Finish capture
            self.stop_capture()
            self.generate_instruction_halt()
    
            # Write captured instructions to DRAM
            program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
            
            # Allocate program DRAM if we auto-allocated the address
            if auto_allocated:
                self.allocate_program_dram(program_size_bytes)
    
            # ===== Print debug information =====
            print(f"\n{'='*80}")
            print(f"[batched_matmat_mul{op_suffix}] Instruction Program Summary")
            print(f"{'='*80}")
            print(f"Declaration: M={M}, K={K}, N={N}, batch_size={batch_size}")
            print(f"  - input_dram_addr_batch:  {hex(input_dram_addr_batch)}")
            print(f"  - weight_dram_addr_batch: {hex(weight_dram_addr_batch)}")
            print(f"  - output_dram_addr_batch: {hex(output_dram_addr_batch)}")
            if bias_enable:
                print(f"  - bias_dram_addr_batch:  {hex(bias_dram_addr_batch)}")
            print(f"  - bias_enable: {bias_enable}, bias_matrix_enable: {bias_matrix_enable}")
            print(f"  - softmax_enable: {softmax_enable}, gelu_enable: {gelu_enable}, silu_enable: {silu_enable}")
            if softmax_enable:
                print(f"\nSoftmax Loop Statistics:")
                print(f"  - Outer loop (while remaining_M_vectors > 0): {outer_loop_count} iterations (x)")
                print(f"  - For loop (for i in range(current_max_M_chunk)): {for_loop_total} total iterations (y)")
                print(f"  - Inner loop (while remaining_N_chunks > 0): {inner_loop_total} total iterations (z)")
                expected_inst_count = 5 + 2 + 3 * outer_loop_count + 5 * for_loop_total + 5 * inner_loop_total + 6 + 1
                expected_unroll_inst_count = 5 + (2 + 3 * outer_loop_count + 5 * for_loop_total + 5 * inner_loop_total + 6) * batch_size + 1
                print(f"  - Expected instruction count: 5 + 2 + 3x + 5y + 5z + 6 + 1 = {expected_inst_count}")
                print(f"  - Expected unrolled instruction count: 5 + (2 + 3x + 5y + 5z + 6) * b + 1 = {expected_unroll_inst_count}")
    
            num_to_print = min(10, self.capture_count)
            if num_to_print > 0:
                print(f"\nFirst {num_to_print} instruction(s):")
                for i in range(num_to_print):
                    inst = self.capture_buffer[i]
                    inst_addr = program_dram_addr + i * 32
                    self.parse_instruction(inst, i, inst_addr)
    
            print(f"{'='*80}\n")
    
            def handler(input_matrix, weight_matrix, bias = None):
                """
                Run batched BF16 matrix-matrix multiply.
                """
                is_input_device = isinstance(input_matrix, DeviceTensor)
                is_weight_device = isinstance(weight_matrix, DeviceTensor)
    
                # Handle Input DMA
                if is_input_device:
                    input_data = input_matrix._data
                    if not input_matrix.needs_dma(input_dram_addr_batch):
                        pass # Cache hit
                    else:
                        input_matrix.sync(input_dram_addr_batch)
                else:
                    input_data = input_matrix
                    self.dma_write(DMA_DEVICE_H2C, input_dram_addr_batch, input_data.flatten(), input_batch_stride * batch_size)
    
                # Handle Weight DMA
                if is_weight_device:
                    weight_data = weight_matrix._data
                    if not weight_matrix.needs_dma(weight_dram_addr_batch):
                        pass # Cache hit
                    else:
                        weight_matrix.sync(weight_dram_addr_batch)
                else:
                    weight_data = weight_matrix
                    self.dma_write(DMA_DEVICE_H2C, weight_dram_addr_batch, weight_data.contiguous().flatten(), weight_batch_stride * batch_size)
    
                # Handle Bias DMA
                if bias_enable is True:
                    assert bias is not None
                    if isinstance(bias, DeviceTensor):
                        if bias.needs_dma(bias_dram_addr_batch):
                            bias.sync(bias_dram_addr_batch)
                    else:
                        self.dma_write(DMA_DEVICE_H2C, bias_dram_addr_batch, bias.contiguous().flatten(), bias_batch_stride * batch_size)
    
                # Execute
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
                self.report_timing_and_instruction_count()
                
                # Flop Reporting
                total_flops = 2 * M * K * N * batch_size
                if bias_enable: total_flops += M * N * batch_size
                if softmax_enable: total_flops += M * N * 5 * batch_size
                elif gelu_enable or silu_enable: total_flops += M * N * 8 * batch_size
    
                print(f"[batched_matmat] {self.report_flop_rate_gflops(total_flops):.2f} GFLOPS")
    
                output_tensor = DeviceTensor((batch_size, M, N), ue=self, dram_addr=output_dram_addr_batch)
                if is_input_device or is_weight_device:
                    return output_tensor
                else:
                    return output_tensor.data
    
            return handler
    
        def batched_matmat_mul_old(self, M: int, K: int, N: int,
                                        input_dram_addr_batch: int,
                                        weight_dram_addr_batch: int,
                                        output_dram_addr_batch: int,
                                        batch_size: int = 1,
                                        program_dram_addr: Optional[int] = None,
                                        bias_enable: bool = False,
                                        bias_matrix_enable: bool = False,
                                        bias_dram_addr_batch: Optional[int] = None,
                                        softmax_enable: bool = False,
                                        gelu_enable: bool = False,
                                        silu_enable: bool = False) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
            """
            Create a reusable Batched BF16 matrix-matrix multiplication.
            
            Computes: C = A @ B.T [+ GELU/SILU/Softmax] for a batch of inputs.
            
            Args:
                M, K, N: Matrix dimensions (M=Rows, K=Inner, N=Cols)
                input_dram_addr_batch: Base DRAM address for Input Batch
                weight_dram_addr_batch: Base DRAM address for Weight Batch
                output_dram_addr_batch: Base DRAM address for Output Batch
                batch_size: Number of matrix multiplications to perform
            """
            bytes_per_element = 2
    
            # Verify alignment
            if K % UE_VECTOR_SIZE != 0:
                raise ValueError(f"K={K} must be a multiple of {UE_VECTOR_SIZE}")
            if N % UE_VECTOR_SIZE != 0:
                raise ValueError(f"N={N} must be a multiple of {UE_VECTOR_SIZE}")
    
            if bias_enable is True:
                assert bias_dram_addr_batch is not None, "bias_dram_addr_batch must be provided when bias_enable is True"
                if bias_matrix_enable is True:
                    total_bias_bytes = M * N * bytes_per_element
                else:
                    total_bias_bytes = N * bytes_per_element
    
            # Validate mutually exclusive options
            activation_count = sum([gelu_enable, silu_enable, softmax_enable])
            if activation_count > 1:
                raise ValueError("gelu_enable, silu_enable, and softmax_enable are mutually exclusive")
    
            op_suffix = ""
            if gelu_enable: op_suffix = " + GELU"
            elif silu_enable: op_suffix = " + SILU"
            elif softmax_enable: op_suffix = " + Softmax"
    
            print(f"[batched_matmat_mul{op_suffix}] Batch={batch_size}, M={M}, K={K}, N={N}")
    
            # ===== Calculate total bytes per batch item =====
            input_batch_stride = M * K * bytes_per_element
            weight_batch_stride = K * N * bytes_per_element
            output_batch_stride = M * N * bytes_per_element
            bias_batch_stride = (M * N if bias_matrix_enable else N) * bytes_per_element
    
            # ===== Capture instructions =====
            self.start_capture()
            inst_id = 0
    
            # Iterate through the batch
            for batch_idx in range(batch_size):
                # Calculate current DRAM offsets
                input_dram_addr = input_dram_addr_batch + batch_idx * input_batch_stride
                weight_dram_addr = weight_dram_addr_batch + batch_idx * weight_batch_stride
                output_dram_addr = output_dram_addr_batch + batch_idx * output_batch_stride
                
                if bias_enable:
                    bias_dram_addr = bias_dram_addr_batch + batch_idx * bias_batch_stride
    
                if softmax_enable:
                    # print("softmax needs to be handled properly in this case!!!")
                    N_ALIGNED = (N // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
    
                    max_N_chunk = ((URAM_NEAR_FULL_ELEMENTS // K) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE 
                    max_N_chunk = min(max_N_chunk, N)
                    
                    max_M_chunk = (((URAM_FULL_ELEMENTS - N_ALIGNED) // K) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
                    output_row_uram_addr = URAM_START_ADDR + K * (max_M_chunk // UE_VECTOR_SIZE)
    
                    max_M_chunk = min(max_M_chunk, M)
                    remaining_M_vectors = M
                    input_chunk_dram_addr = input_dram_addr
                    output_chunk_dram_addr = output_dram_addr
    
                    while remaining_M_vectors > 0:
                        current_max_M_chunk = min(max_M_chunk, remaining_M_vectors)
                        current_M_chunk_bytes = current_max_M_chunk * K * bytes_per_element
    
                        input_uram_addr = URAM_START_ADDR
                        self.ue_memcpy_from_dram(input_chunk_dram_addr, current_M_chunk_bytes, 0, input_uram_addr, URAM_SECTION.URAM_A.value, inst_id)
                        self.wait_queue()
                        inst_id += 1
    
                        for i in range(current_max_M_chunk):
                            remaining_N_chunks = N
                            weight_chunk_dram_addr = weight_dram_addr
                            output_uram_addr = output_row_uram_addr
                            clear_max_en = 1
                            
                            while remaining_N_chunks > 0:
                                current_N_chunk = min(max_N_chunk, remaining_N_chunks)
                                current_N_chunk_bytes = current_N_chunk * K * bytes_per_element
    
                                self.ue_memcpy_from_dram(weight_chunk_dram_addr, current_N_chunk_bytes, 0, URAM_START_ADDR, URAM_SECTION.URAM_B.value, inst_id)
                                self.wait_queue()
                                inst_id += 1
    
                                if bias_enable is True:
                                    if bias_matrix_enable is True:
                                        bias_offset = bias_dram_addr + ((N - remaining_N_chunks) + N * (M - remaining_M_vectors + i)) * bytes_per_element
                                    else:
                                        bias_offset = bias_dram_addr + (N - remaining_N_chunks) * bytes_per_element
                                    self.ue_memcpy_from_dram(bias_offset, current_N_chunk * bytes_per_element, MEMCPY_TYPE.BIAS_BRAM.value, 0, 0, inst_id)
                                    self.wait_queue()
                                    inst_id += 1
    
                                remaining_N_chunks -= current_N_chunk
                                weight_chunk_dram_addr += current_N_chunk_bytes
    
                                self.start_queue(
                                    0, clear_max_en, 1, LALU_MODE.BYPASS.value, 0, 0,
                                    URAM_SECTION.URAM_A.value, 0, 0, output_uram_addr,
                                    URAM_WRITE_SRC.URAM_WRITE_BACK.value, UE_MODE.BF16_DOT_PRODUCT, 0,
                                    input_uram_addr + i * (K // UE_VECTOR_SIZE), URAM_START_ADDR,
                                    K // UE_VECTOR_SIZE, 0, current_N_chunk * K, current_N_chunk,
                                    inst_id, 1 if bias_enable else 0
                                )
                                inst_id += 1
                                self.wait_queue()
                                clear_max_en = 0
                                output_uram_addr += current_N_chunk // UE_VECTOR_SIZE
    
                            # FMAX / SOFTMAX Passes
                            self.start_queue(
                                BROADCAST_MODE.FMAX_NEGATE.value, 0, 1, LALU_MODE.MODE_RECIP.value, 0x9FC00, 0,
                                URAM_SECTION.URAM_A.value, 0, 0, output_row_uram_addr,
                                URAM_WRITE_SRC.URAM_WRITE_BACK.value, UE_MODE.EXP, 0,
                                output_row_uram_addr, 0, N // UE_VECTOR_SIZE, 0, 0, 0, inst_id
                            )
                            inst_id += 1
                            self.wait_queue()
    
                            self.start_queue(
                                BROADCAST_MODE.LALU_RESULT.value, 0, 1, LALU_MODE.BYPASS.value, 0, 0,
                                URAM_SECTION.URAM_A.value, 0, 0, output_row_uram_addr,
                                URAM_WRITE_SRC.URAM_WRITE_BACK.value, UE_MODE.MUL_BROADCAST, 0,
                                output_row_uram_addr, 0, N // UE_VECTOR_SIZE, 0, 0, 0, inst_id
                            )
                            inst_id += 1
                            self.wait_queue()
    
                            self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, output_row_uram_addr, output_chunk_dram_addr + i * N * bytes_per_element, N * bytes_per_element, inst_id)
                            self.wait_queue()
                            inst_id += 1
    
                        remaining_M_vectors -= current_max_M_chunk
                        output_chunk_dram_addr += current_max_M_chunk * N * bytes_per_element
                        input_chunk_dram_addr += current_M_chunk_bytes
                else:
                    # ===== Standard Path (No Softmax) =====
                    max_N_chunk = ((URAM_NEAR_FULL_ELEMENTS // K) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
                    max_N_chunk = min(max_N_chunk, N)
                    
                    # Optimization logic for M_chunk
                    ratio = (max_N_chunk / K) + 1.0
                    L = int(URAM_FULL_ELEMENTS / ratio)
                    max_M_chunk = min(M, L // K, (URAM_FULL_ELEMENTS - L) // max_N_chunk)
    
                    # Determine LALU mode
                    if gelu_enable:
                        dp_lalu_mode = LALU_MODE.GELU.value
                    elif silu_enable:
                        dp_lalu_mode = LALU_MODE.SILU.value
                    else:
                        dp_lalu_mode = LALU_MODE.BYPASS.value
                    dp_lalu_scalar = 0
    
                    remaining_M_vectors = M
                    input_chunk_dram_addr = input_dram_addr
                    output_chunk_dram_addr = output_dram_addr
                    
                    while remaining_M_vectors > 0:
                        current_max_M_chunk = min(max_M_chunk, remaining_M_vectors)
                        current_M_chunk_bytes = current_max_M_chunk * K * bytes_per_element
    
                        input_uram_addr = URAM_START_ADDR
                        output_uram_addr = L // UE_VECTOR_SIZE
                        
                        # Load Input Chunk
                        self.ue_memcpy_from_dram(input_chunk_dram_addr, current_M_chunk_bytes, 0, input_uram_addr, URAM_SECTION.URAM_A.value, inst_id)
                        self.wait_queue()
                        inst_id += 1
    
                        remaining_N_chunks = N
                        weight_chunk_dram_addr = weight_dram_addr
                        output_chunk_dram_offset_in_chunk_row = output_chunk_dram_addr
                        
                        while remaining_N_chunks > 0:
                            current_N_chunk = min(max_N_chunk, remaining_N_chunks)
                            current_N_chunk_bytes = current_N_chunk * K * bytes_per_element
    
                            # Load Weight Chunk
                            self.ue_memcpy_from_dram(weight_chunk_dram_addr, current_N_chunk_bytes, 0, URAM_START_ADDR, URAM_SECTION.URAM_B.value, inst_id)
                            self.wait_queue()
                            inst_id += 1
    
                            if bias_enable is True:
                                if bias_matrix_enable is False:
                                    bias_offset = bias_dram_addr + (N - remaining_N_chunks) * bytes_per_element
                                    self.ue_memcpy_from_dram(bias_offset, current_N_chunk * bytes_per_element, MEMCPY_TYPE.BIAS_BRAM.value, 0, 0, inst_id)
                                    self.wait_queue()
                                    inst_id += 1
    
                            for i in range(current_max_M_chunk):
                                if bias_enable is True and bias_matrix_enable is True:
                                    bias_offset = bias_dram_addr + ((N - remaining_N_chunks) + N * (M - remaining_M_vectors + i)) * bytes_per_element
                                    self.ue_memcpy_from_dram(bias_offset, current_N_chunk * bytes_per_element, MEMCPY_TYPE.BIAS_BRAM.value, 0, 0, inst_id)
                                    self.wait_queue()
                                    inst_id += 1
    
                                self.start_queue(
                                    0, 0, 1, dp_lalu_mode, dp_lalu_scalar, 0,
                                    URAM_SECTION.URAM_A.value, 0, 0, output_uram_addr,
                                    URAM_WRITE_SRC.URAM_WRITE_BACK.value, UE_MODE.BF16_DOT_PRODUCT, 0,
                                    input_uram_addr + i * (K // UE_VECTOR_SIZE), URAM_START_ADDR,
                                    K // UE_VECTOR_SIZE, 0, current_N_chunk * K, current_N_chunk,
                                    inst_id, 1 if bias_enable else 0
                                )
                                inst_id += 1
                                self.wait_queue()
    
                                self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, output_uram_addr, output_chunk_dram_offset_in_chunk_row + i * N * bytes_per_element, current_N_chunk * bytes_per_element, inst_id)
                                self.wait_queue()
                                inst_id += 1
    
                            remaining_N_chunks -= current_N_chunk
                            output_chunk_dram_offset_in_chunk_row += current_N_chunk * bytes_per_element
                            weight_chunk_dram_addr += current_N_chunk_bytes
    
                        output_chunk_dram_addr += current_max_M_chunk * N * bytes_per_element
                        remaining_M_vectors -= current_max_M_chunk
                        input_chunk_dram_addr += current_M_chunk_bytes
    
            # Finish capture
            self.stop_capture()
            self.generate_instruction_halt()
    
            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
                self.allocate_program_dram(program_size_bytes)
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
                self.write_captured_instructions_to_dram(program_dram_addr)
    
            def handler(input_matrix, weight_matrix, bias = None):
                """
                Run batched BF16 matrix-matrix multiply.
                """
                is_input_device = isinstance(input_matrix, DeviceTensor)
                is_weight_device = isinstance(weight_matrix, DeviceTensor)
    
                # Handle Input DMA
                if is_input_device:
                    input_data = input_matrix._data
                    if not input_matrix.needs_dma(input_dram_addr_batch):
                        pass # Cache hit
                    else:
                        input_matrix.sync(input_dram_addr_batch)
                else:
                    input_data = input_matrix
                    self.dma_write(DMA_DEVICE_H2C, input_dram_addr_batch, input_data.flatten(), input_batch_stride * batch_size)
    
                # Handle Weight DMA
                if is_weight_device:
                    weight_data = weight_matrix._data
                    if not weight_matrix.needs_dma(weight_dram_addr_batch):
                        pass # Cache hit
                    else:
                        weight_matrix.sync(weight_dram_addr_batch)
                else:
                    weight_data = weight_matrix
                    self.dma_write(DMA_DEVICE_H2C, weight_dram_addr_batch, weight_data.contiguous().flatten(), weight_batch_stride * batch_size)
    
                # Handle Bias DMA
                if bias_enable is True:
                    assert bias is not None
                    if isinstance(bias, DeviceTensor):
                        if bias.needs_dma(bias_dram_addr_batch):
                            bias.sync(bias_dram_addr_batch)
                    else:
                        self.dma_write(DMA_DEVICE_H2C, bias_dram_addr_batch, bias.contiguous().flatten(), bias_batch_stride * batch_size)
    
                # Execute
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
                self.report_timing_and_instruction_count()
                
                # Flop Reporting
                total_flops = 2 * M * K * N * batch_size
                if bias_enable: total_flops += M * N * batch_size
                if softmax_enable: total_flops += M * N * 5 * batch_size
                elif gelu_enable or silu_enable: total_flops += M * N * 8 * batch_size
    
                print(f"[batched_matmat] {self.report_flop_rate_gflops(total_flops):.2f} GFLOPS")
    
                output_tensor = DeviceTensor((batch_size, M, N), ue=self, dram_addr=output_dram_addr_batch)
                if is_input_device or is_weight_device:
                    return output_tensor
                else:
                    return output_tensor.data
    
            return handler
    
        def matmat_mul(self, M: int, K: int, N: int,
                                                input_dram_addr: int,
                                                weight_dram_addr: int,
                                                output_dram_addr: int,
                                                program_dram_addr: Optional[int] = None,
                                                bias_enable: bool = False,
                                                bias_matrix_enable: bool = False,
                                                bias_dram_addr: Optional[int] = None,
                                                softmax_enable: bool = False,
                                                gelu_enable: bool = False,
                                                silu_enable: bool = False) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
            """
            Create a reusable BF16 matrix-matrix multiplication with optimized M and N partitioning.
    
            Computes: C = A @ B.T [+ GELU/SILU], where both input and weight matrices are provided
            per call (both are activations, not pre-stored weights).
    
            Where:
                A: Input activation matrix (M x K) in bf16 - provided per call
                B: Weight matrix (N x K) in bf16 - provided per call, transpose handled by hardware
                C: Output matrix (M x N) in bf16
    
            Comparison with bf16_matmat_activation:
            =======================================
            - bf16_matmat_activation: Partitions primarily by N, loads input once per N-chunk.
              Weights are stored during setup. Supports softmax.
            - bf16_matmat_activation_k_partition: Partitions by both M and N, reloads input for
              each N-chunk. Both matrices provided per call. Better for large K when input
              cannot fit in URAM once. Does NOT support softmax.
    
            Dimension Convention:
            =====================
                M = batch/sequence dimension (number of input vectors)
                K = input features (shared dimension, must be multiple of 64)
                N = output features (must be multiple of 64)
    
            Partitioning Strategy:
            ======================
            The function optimizes URAM usage by dynamically splitting URAM_A between input
            storage and output accumulation:
    
            1. N_chunk = min(N, (URAM_NEAR_FULL // K) // 64 * 64)
               - Maximum weight rows that fit in URAM_B
               - Rounded down to multiple of 64
    
            2. M_chunk calculation uses optimized split:
               - L = URAM_FULL / ((N_chunk / K) + 1.0)
               - M_chunk = min(M, L // K, (URAM_FULL - L) // N_chunk)
               - This balances input storage (L elements) vs output accumulation space
    
            URAM Memory Map:
            ================
            URAM_A:                              URAM_B:
            ┌─────────────────────────┐          ┌─────────────────────────┐
            │ 0x000                   │          │ 0x000                   │
            │ A_chunk (M_chunk x K)   │ ◄─Input  │ B_chunk (N_chunk x K)   │ ◄─Weight chunk
            │ (first L elements)      │          │ (loaded per iteration)  │
            └─────────────────────────┘          └─────────────────────────┘
            │ L/64 (HALFWAY)          │
            │ C_partial (1 x N_chunk) │ ◄─Output (one row at a time)
            └─────────────────────────┘
    
            Bias Adder:
            ============
            - when BIAS_BRAM is loaded and bias adder is enabled
            - Bias is added to the result for all vectors in this batch
    
            Processing Flow:
            ================
            For each M_chunk of input rows:
              1. Load A_chunk (M_chunk x K) to URAM_A[0x000]
              For each N_chunk of weight rows:
                2. Load B_chunk (N_chunk x K) to URAM_B[0x000]
                For each row i in M_chunk:
                  3. BF16_DOT_PRODUCT: A[i, :] @ B_chunk.T → C[i, N_chunk_range]
                     - Input vector: URAM_A[i * (K/64)]
                     - Weight chunk: URAM_B[0x000]
                     - Output: URAM_A[L/64] (single row result)
                  4. Write C[i, N_chunk_range] to DRAM at offset (i * N + N_chunk_start)
    
            When to Use:
            ============
            - Large K dimension where input cannot fit in URAM once
            - Variable input/weight matrices provided per call (not pre-stored)
            - When bf16_matmat_activation runs into memory constraints
            - Attention-like operations with variable sequence lengths
            - When softmax is NOT needed (use bf16_matmat_activation for softmax)
    
            Args:
                M: Batch/sequence dimension (number of input vectors)
                K: Input features (must be multiple of 64)
                N: Output features (must be multiple of 64)
                input_dram_addr: DRAM address for input matrix (M x K) - provided per call
                weight_dram_addr: DRAM address for weight matrix (N x K) - provided per call
                output_dram_addr: DRAM address for output matrix (M x N)
                program_dram_addr: DRAM address for instruction stream. If None, auto-allocated.
                bias_enable: Enable bias addition (currently not implemented, reserved for future)
                gelu_enable: Enable GELU activation after multiply (default: False)
                silu_enable: Enable SILU activation after multiply (default: False)
    
            Returns:
                handler: Callable that takes (input_matrix: torch.Tensor, weight_matrix: torch.Tensor)
                         and returns output matrix (M, N) in bf16 format. Both input and weight
                         matrices must match the dimensions specified during setup.
            """
            bytes_per_element = 2
    
            # Verify alignment
            if K % UE_VECTOR_SIZE != 0:
                raise ValueError(f"K={K} must be a multiple of {UE_VECTOR_SIZE}")
            if N % UE_VECTOR_SIZE != 0:
                raise ValueError(f"N={N} must be a multiple of {UE_VECTOR_SIZE}")
    
            if bias_enable is True:
                assert bias_dram_addr is not None, "bias_dram_addr must be provided when bias_enable is True"
                if bias_matrix_enable is True:
                    total_bias_bytes = M * N * bytes_per_element
                else:
                    total_bias_bytes = N * bytes_per_element
    
            # Validate mutually exclusive options
            activation_count = sum([gelu_enable, silu_enable, softmax_enable])
            if activation_count > 1:
                raise ValueError("gelu_enable, silu_enable, and softmax_enable are mutually exclusive")
    
            op_suffix = ""
            if gelu_enable:
                op_suffix = " + GELU"
            elif silu_enable:
                op_suffix = " + SILU"
            elif softmax_enable:
                op_suffix = " + Softmax"
    
            print(f"[bf16_matmat_activation_k_partition{op_suffix}] M={M}, K={K}, N={N}")
    
            # ===== Calculate total bytes =====
            total_input_bytes = M * K * bytes_per_element
            total_weight_bytes = K * N * bytes_per_element
    
            # ===== Capture instructions =====
            self.start_capture()
            inst_id = 0
    
            if softmax_enable:
                # print("softmax needs to be handled properly in this case!!!")
                N_ALIGNED = (N // UE_VECTOR_SIZE) * UE_VECTOR_SIZE
    
                max_N_chunk = ((URAM_NEAR_FULL_ELEMENTS // K) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE # Lower to nearest multiple of 64
                max_N_chunk = min(max_N_chunk, N) # matrix is loaded into URAM_B with chunks of K * max_N_chunk elements
                assert max_N_chunk >= 1 and max_N_chunk <= N, f"max_N_chunk={max_N_chunk} must be greater than 0 and less than N={N}"
    
                max_M_chunk = (((URAM_FULL_ELEMENTS - N_ALIGNED) // K) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE # Lower to nearest multiple of 64
                output_row_uram_addr = URAM_START_ADDR + K * (max_M_chunk // UE_VECTOR_SIZE)
    
                max_M_chunk = min(max_M_chunk, M)
                assert max_M_chunk >= 1 and max_M_chunk <= M, f"max_M_chunk={max_M_chunk} must be greater than 0 and less than M={M}"
    
                remaining_M_vectors = M
                input_chunk_dram_addr = input_dram_addr
                output_chunk_dram_addr = output_dram_addr
    
                while remaining_M_vectors > 0:
                    current_max_M_chunk = min(max_M_chunk, remaining_M_vectors)
                    current_M_chunk_bytes = current_max_M_chunk * K * bytes_per_element
    
                    input_uram_addr = URAM_START_ADDR
                    # Load max_M_chunk x K elements from DRAM to URAM_A
                    self.ue_memcpy_from_dram(input_chunk_dram_addr, current_M_chunk_bytes, 0, input_uram_addr, URAM_SECTION.URAM_A.value, inst_id)
                    self.wait_queue()
                    inst_id += 1
    
                    for i in range(current_max_M_chunk):
    
                        remaining_N_chunks = N
                        weight_chunk_dram_addr = weight_dram_addr
                        output_uram_addr = output_row_uram_addr
    
                        clear_max_en = 1
                        # weight matrix chunking is done here
                        while remaining_N_chunks > 0:
                            current_N_chunk = min(max_N_chunk, remaining_N_chunks)
                            current_N_chunk_bytes = current_N_chunk * K * bytes_per_element
    
                            # largest possible weight matrix chunk is copied to URAM_B
                            self.ue_memcpy_from_dram(weight_chunk_dram_addr, current_N_chunk_bytes, 0, URAM_START_ADDR, URAM_SECTION.URAM_B.value, inst_id)
                            self.wait_queue()
                            inst_id += 1
    
                            if bias_enable is True:
                                if bias_matrix_enable is True   :
                                    bias_offset = bias_dram_addr + ((N - remaining_N_chunks) + N * (M - remaining_M_vectors + i)) * bytes_per_element
                                else:
                                    bias_offset = bias_dram_addr + (N - remaining_N_chunks) * bytes_per_element
                                self.ue_memcpy_from_dram(bias_offset, current_N_chunk * bytes_per_element, MEMCPY_TYPE.BIAS_BRAM.value, 0, 0, inst_id)
                                self.wait_queue()
                                inst_id += 1
    
                            remaining_N_chunks -= current_N_chunk
                            weight_chunk_dram_addr += current_N_chunk_bytes # move weight chunk dram address to next chunk
    
                            # BF16 dot product: x_vector @ w.T
                            self.start_queue(
                                0,  # broadcast_mode
                                clear_max_en,  # clear_max_en
                                1,  # stride_z
                                LALU_MODE.BYPASS.value,  # lalu_mode
                                0,  # lalu_scalar
                                0,  # uram_bram
                                URAM_SECTION.URAM_A.value,  # uram_section
                                0,  # uram_dst_addr not used
                                0,  # dram_to_uram_cpy_start not used
                                output_uram_addr,  # uram_wb_addr
                                URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                                UE_MODE.BF16_DOT_PRODUCT,  # mode
                                0,  # data_type
                                input_uram_addr + i * (K // UE_VECTOR_SIZE),  # uram_a_start_addr
                                URAM_START_ADDR,  # uram_b_start_addr
                                K // UE_VECTOR_SIZE,  # uram_length
                                0,  # dma_start_addr not used
                                current_N_chunk * K,  # dma_length
                                current_N_chunk,  # output_size
                                inst_id,
                                1 if bias_enable is True else 0
                            )
                            inst_id += 1
                            self.wait_queue()
    
                            clear_max_en = 0 # clear to generate pulse
    
                            output_uram_addr += current_N_chunk // UE_VECTOR_SIZE
    
                        # FMAX is here
                        # Pass 1: EXP mode with FMAX_NEGATE
                        # exp(y - max(y)) → softmax_output_addr, 1/sum → LALU
                        self.start_queue(
                            BROADCAST_MODE.FMAX_NEGATE.value,  # broadcast_mode
                            0,  # clear_max_en
                            1,  # stride_z
                            LALU_MODE.MODE_RECIP.value,  # lalu_mode
                            0x9FC00,  # lalu_scalar
                            0,  # uram_bram
                            URAM_SECTION.URAM_A.value,  # uram_section
                            0,  # uram_dst_addr
                            0,  # dram_to_uram_cpy_start
                            output_row_uram_addr,  # uram_wb_addr
                            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                            UE_MODE.EXP,  # mode
                            0,  # data_type
                            output_row_uram_addr,  # uram_a_start_addr
                            0,  # uram_b_start_addr
                            N // UE_VECTOR_SIZE,  # uram_length
                            0,  # dma_start_addr
                            0,  # dma_length
                            0,  # output_size
                            inst_id
                        )
                        inst_id += 1
                        self.wait_queue()
    
                        # Pass 2: MUL_BROADCAST mode
                        # exp_result * (1/sum) → softmax_output_addr
                        self.start_queue(
                            BROADCAST_MODE.LALU_RESULT.value,  # broadcast_mode
                            0,  # clear_max_en
                            1,  # stride_z
                            LALU_MODE.BYPASS.value,  # lalu_mode
                            0,  # lalu_scalar
                            0,  # uram_bram
                            URAM_SECTION.URAM_A.value,  # uram_section
                            0,  # uram_dst_addr
                            0,  # dram_to_uram_cpy_start
                            output_row_uram_addr,  # uram_wb_addr
                            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                            UE_MODE.MUL_BROADCAST,  # mode
                            0,  # data_type
                            output_row_uram_addr,  # uram_a_start_addr
                            0,  # uram_b_start_addr
                            N // UE_VECTOR_SIZE,  # uram_length
                            0,  # dma_start_addr
                            0,  # dma_length
                            0,  # output_size
                            inst_id
                        )
                        inst_id += 1
                        self.wait_queue()
    
                        self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, output_row_uram_addr, output_chunk_dram_addr + i * N * bytes_per_element, N * bytes_per_element, inst_id)
                        self.wait_queue()
                        inst_id += 1
    
                    remaining_M_vectors -= current_max_M_chunk
                    output_chunk_dram_addr += current_max_M_chunk * N * bytes_per_element
                    input_chunk_dram_addr += current_M_chunk_bytes
            else:
                # ===== Calculate URAM_B partitioning =====
                # Maximum N_chunk that fits in URAM_B with K columns
                max_N_chunk = ((URAM_NEAR_FULL_ELEMENTS // K) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE # Lower to nearest multiple of 64
                max_N_chunk = min(max_N_chunk, N) # matrix is loaded into URAM_B with chunks of K * max_N_chunk elements
                assert max_N_chunk >= 1 and max_N_chunk <= N, f"max_N_chunk={max_N_chunk} must be greater than 0 and less than N={N}"
    
                max_M_chunk = min(M, URAM_HALF_ELEMENTS // K, URAM_HALF_ELEMENTS // max_N_chunk)
                print(f"max_M_chunk before optimization={max_M_chunk}")
                ratio = (max_N_chunk / K) + 1.0
                L = int(URAM_FULL_ELEMENTS / ratio)
                print(f"L={L} and K={K}")
                print(f"R={URAM_FULL_ELEMENTS - L} and max_N_chunk={max_N_chunk}")
                max_M_chunk = min(M, L // K, (URAM_FULL_ELEMENTS - L) // max_N_chunk)
                print(f"max_M_chunk after optimization={max_M_chunk}")
    
                assert max_M_chunk >= 1 and max_M_chunk <= M, f"max_M_chunk={max_M_chunk} must be greater than 0 and less than M={M}"
    
                print(f"URAMB can have: {max_N_chunk}x{K} of {N}x{K}")
    
                # Uram A first half is max_M_chunk x K and second half is K x max_N_chunk
                # Max calculated matrices by hw is thus (max_M_chunk x K) and (K x max_N_chunk)
                # iterations are 0:max_M_chunk:M and 0:max_N_chunk:N
    
                # Determine LALU mode for dot product (GELU/SILU only if no softmax)
                if gelu_enable:
                    dp_lalu_mode = LALU_MODE.GELU.value
                    dp_lalu_scalar = 0
                elif silu_enable:
                    dp_lalu_mode = LALU_MODE.SILU.value
                    dp_lalu_scalar = 0
                else:
                    dp_lalu_mode = LALU_MODE.BYPASS.value
                    dp_lalu_scalar = 0
    
                # Process each weight chunk
                remaining_M_vectors = M
                input_chunk_dram_addr = input_dram_addr
                output_chunk_dram_addr = output_dram_addr
                while remaining_M_vectors > 0:
                    current_max_M_chunk = min(max_M_chunk, remaining_M_vectors)
                    current_M_chunk_bytes = current_max_M_chunk * K * bytes_per_element
    
                    input_uram_addr = URAM_START_ADDR
                    output_uram_addr = L // UE_VECTOR_SIZE
                    # Load max_M_chunk x K elements from DRAM to URAM_A
                    self.ue_memcpy_from_dram(input_chunk_dram_addr, current_M_chunk_bytes, 0, input_uram_addr, URAM_SECTION.URAM_A.value, inst_id)
                    self.wait_queue()
                    inst_id += 1
    
                    remaining_N_chunks = N
                    weight_chunk_dram_addr = weight_dram_addr
    
                    # weight matrix chunking is done here
                    output_chunk_dram_offset_in_chunk_row = output_chunk_dram_addr
                    while remaining_N_chunks > 0:
                        current_N_chunk = min(max_N_chunk, remaining_N_chunks)
                        current_N_chunk_bytes = current_N_chunk * K * bytes_per_element
    
                        # weight is always copied to URAM_B from start of URAM_B
                        self.ue_memcpy_from_dram(weight_chunk_dram_addr, current_N_chunk_bytes, 0, URAM_START_ADDR, URAM_SECTION.URAM_B.value, inst_id)
                        self.wait_queue()
                        inst_id += 1
    
                        if bias_enable is True:
                            if bias_matrix_enable is False:
                                bias_offset = bias_dram_addr + (N - remaining_N_chunks) * bytes_per_element
                                self.ue_memcpy_from_dram(bias_offset, current_N_chunk * bytes_per_element, MEMCPY_TYPE.BIAS_BRAM.value, 0, 0, inst_id)
                                self.wait_queue()
                                inst_id += 1
    
                        # calculate one chunk of output for each input vector
                        for i in range(current_max_M_chunk):
    
                            if bias_enable is True:
                                if bias_matrix_enable is True:
                                    bias_offset = bias_dram_addr + ((N - remaining_N_chunks) + N * (M - remaining_M_vectors + i)) * bytes_per_element
                                    self.ue_memcpy_from_dram(bias_offset, current_N_chunk * bytes_per_element, MEMCPY_TYPE.BIAS_BRAM.value, 0, 0, inst_id)
                                    self.wait_queue()
                                    inst_id += 1
    
                            # BF16 dot product
                            self.start_queue(
                                0,  # broadcast_mode
                                0,  # clear_max_en not used
                                1,  # stride_z
                                dp_lalu_mode,  # lalu_mode
                                dp_lalu_scalar,  # lalu_scalar
                                0,  # uram_bram
                                URAM_SECTION.URAM_A.value,  # uram_section
                                0,  # uram_dst_addr
                                0,  # dram_to_uram_cpy_start
                                output_uram_addr,  # uram_wb_addr
                                URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                                UE_MODE.BF16_DOT_PRODUCT,  # mode
                                0,  # data_type
                                input_uram_addr + i * (K // UE_VECTOR_SIZE),  # uram_a_start_addr
                                URAM_START_ADDR,  # uram_b_start_addr
                                K // UE_VECTOR_SIZE,  # uram_length
                                0,  # dma_start_addr not used
                                current_N_chunk * K,  # dma_length
                                current_N_chunk,  # output_size
                                inst_id,
                                1 if bias_enable is True else 0
                            )
                            inst_id += 1
                            self.wait_queue()
    
                            # TODO: Optimize here for better mem-copying
                            self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, output_uram_addr, output_chunk_dram_offset_in_chunk_row + i * N * bytes_per_element, current_N_chunk * bytes_per_element, inst_id)
                            self.wait_queue()
                            inst_id += 1
    
                        remaining_N_chunks -= current_N_chunk
                        output_chunk_dram_offset_in_chunk_row += current_N_chunk * bytes_per_element
                        weight_chunk_dram_addr += current_N_chunk_bytes # move weight chunk dram address to next chunk
    
                    output_chunk_dram_addr += current_max_M_chunk * N * bytes_per_element
                    remaining_M_vectors -= current_max_M_chunk
                    input_chunk_dram_addr += current_M_chunk_bytes
    
            # Finish capture and write instruction stream to DRAM
            self.stop_capture()
            self.generate_instruction_halt()
    
            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
                self.allocate_program_dram(program_size_bytes)
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
                self.write_captured_instructions_to_dram(program_dram_addr)
    
            def handler(input_matrix, weight_matrix, bias_vector = None):
                """
                Run BF16 matrix-matrix multiply: Y = X @ W.T [+ GELU/Softmax]
                Both input and weight matrices are provided per call (both are activations).
                Optional bias matrix is provided per call (bias is added to the result).
    
                Supports DeviceTensor inputs for DMA cache optimization - skips DMA if data
                is already synced at the target address.
    
                Args:
                    input_matrix: Input (M, K) - torch.Tensor or DeviceTensor
                    weight_matrix: Weight (N, K) - torch.Tensor or DeviceTensor
                    bias_vector: Bias (N,) - torch.Tensor or DeviceTensor (optional)
    
                Returns:
                    DeviceTensor if any input was DeviceTensor, else raw tensor (M, N)
                """
                # Extract tensor data and check if DMA can be skipped
                is_input_device = isinstance(input_matrix, DeviceTensor)
                is_weight_device = isinstance(weight_matrix, DeviceTensor)
    
                if is_input_device:
                    input_data = input_matrix._data
                    skip_input_dma = not input_matrix.needs_dma(input_dram_addr)
                else:
                    input_data = input_matrix
                    skip_input_dma = False
                    assert input_data.dtype == torch.bfloat16, "Input matrix must be in bf16 format"
    
                if is_weight_device:
                    weight_data = weight_matrix._data
                    skip_weight_dma = not weight_matrix.needs_dma(weight_dram_addr)
                else:
                    weight_data = weight_matrix
                    skip_weight_dma = False
                    assert weight_data.dtype == torch.bfloat16, "Weight matrix must be in bf16 format"
    
                # TODO: Uncomment this when we have a way to validate the input and weight matrices
                # if input_data.dim() == 1:
                #     input_data = input_data.unsqueeze(0)
    
                # input_M, input_K = input_data.shape
                # if input_M != M:
                #     raise ValueError(f"Input matrix has {input_M} rows, expected M={M}")
                # if input_K != K:
                #     raise ValueError(f"Input matrix has {input_K} columns, expected K={K}")
    
                # weight_N, weight_K = weight_data.shape
                # if weight_K != K:
                #     raise ValueError(f"Weight matrix has {weight_K} rows, expected K={K}")
                # if weight_N != N:
                #     raise ValueError(f"Weight matrix has {weight_N} columns, expected N={N}")
    
                # DMA input and weight to DRAM (skip if DeviceTensor is already synced)
                if skip_input_dma:
                    print(f"[DMA cache hit] Skipping input DMA to {hex(input_dram_addr)}")
                else:
                    if isinstance(input_matrix, DeviceTensor):
                        print(f"[DMA cache miss] Input DMA to {hex(input_dram_addr)}")
                        input_matrix.sync(input_dram_addr)
                    else:
                        self.dma_write(DMA_DEVICE_H2C, input_dram_addr, input_data.flatten(), total_input_bytes)
    
                if skip_weight_dma:
                    print(f"[DMA cache hit] Skipping weight DMA to {hex(weight_dram_addr)}")
                else:
                    if isinstance(weight_matrix, DeviceTensor):
                        print(f"[DMA cache miss] Weight DMA to {hex(weight_dram_addr)}")
                        weight_matrix.sync(weight_dram_addr)
                    else:
                        self.dma_write(DMA_DEVICE_H2C, weight_dram_addr, weight_data.contiguous().flatten(), total_weight_bytes)
    
                if bias_enable is True:
                    assert bias_vector is not None, "Bias vector must be provided when bias_enable is True"
                    if isinstance(bias_vector, DeviceTensor):
                        if not bias_vector.needs_dma(bias_dram_addr):
                            print(f"[DMA cache hit] Skipping bias DMA to {hex(bias_dram_addr)}")
                        else:
                            bias_vector.sync(bias_dram_addr)
                            print(f"[DMA cache miss] Bias DMA to {hex(bias_dram_addr)}")
                    else:
                        self.dma_write(DMA_DEVICE_H2C, bias_dram_addr, bias_vector.contiguous().flatten(), total_bias_bytes)
                        print(f"total_bias_bytes={total_bias_bytes}")
    
                # Start executing from DRAM and wait for completion
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
                self.report_timing_and_instruction_count()
                # Matrix-matrix multiply: 2*M*K*N FLOPs
                total_flops = 2 * M * K * N
                if bias_enable is True:
                    total_flops += M * N
                if softmax_enable:
                    total_flops += M * N * 5  # max(M*N) + sub(M*N) + exp(M*N) + sum(M*N) + div(1) + mul(M*N)
                if gelu_enable or silu_enable:
                    total_flops += M * N * 4  # GELU/SILU approximation
    
                theoretical_flops = 128.0 / CLOCK_CYCLE_TIME_NS
                report_flops = self.report_flop_rate_gflops(total_flops)
                print(f"[bf16_matmat_k_partition] {self.report_latency_in_us():.3f} us, {report_flops:.2f} GFLOPS of {theoretical_flops:.2f} GFLOPS")
                flops_ratio = report_flops / theoretical_flops * 100
                print(f"Theoretical FLOPS / Report FLOPS: {flops_ratio:.2f}%")
    
                # Return DeviceTensor if any input was DeviceTensor, else return raw tensor
                output_tensor = DeviceTensor((M, N), ue=self, dram_addr=output_dram_addr)
                if is_input_device or is_weight_device or (bias_enable is True and isinstance(bias_vector, DeviceTensor)):
                    return output_tensor
                else:
                    return output_tensor.data
    
            return handler

        def bf16_transpose(self, M: int, N: int,
                         input_dram_addr: int,
                         output_dram_addr: int,
                         program_dram_addr: Optional[int] = None,
                         params_dram_addr: Optional[int] = None) -> Callable[[torch.Tensor], torch.Tensor]:
            """
            Create a reusable BF16 matrix transpose operation using identity matrix selection.

            Transposes an (M x N) input matrix X to produce an (N x M) output matrix Y = X^T.
            The operation leverages hardware BF16 dot product by using an identity matrix to
            selectively extract rows from the input, which become columns in the output.

            Algorithm:
            ==========
            For each output column j (0 ≤ j < M):
              1. Select the j-th row from the identity matrix I (N x N)
              2. Compute dot product: Y[:, j] = X[j, :] @ I[j, :]^T
              3. This extracts row j from X and places it as column j in Y

            The identity matrix (64x64) is stored once in params_dram_addr during setup and
            reused for all transpose operations.

            Memory Layout:
            ==============
            URAM_A:                              URAM_B:
            ┌─────────────────────────┐          ┌─────────────────────────┐
            │ 0x000                   │          │ 0x000                   │
            │ I (64x64 identity)      │ ◄─Setup  │ X chunk (N_chunk x N)   │ ◄─Input chunk
            │ (stored at setup)       │          │ (loaded per iteration)  │
            │                         │          └─────────────────────────┘
            │ 0x040 (64 elements)     │
            │ Y chunk (1 x N_chunk)   │ ◄─Result
            └─────────────────────────┘

            Chunking Strategy:
            ==================
            Large matrices are processed in chunks to fit within URAM capacity:
            - Input rows: Processed in groups of M_chunk rows (limited by URAM_B capacity)
            - Input columns: Processed in groups of N_chunk columns (limited by URAM_B capacity)
            - For each row i in current M_chunk:
                * Load N_chunk columns of row i into URAM_B
                * Select identity vector for row i from URAM_A
                * Compute dot product to extract row i, column chunk
                * Write result to output DRAM at position (column_chunk, row i)

            Processing Flow:
            ================
            1. Setup: Store 64x64 identity matrix in params_dram_addr and load to URAM_A
            2. For each M_chunk rows:
               a. For each N_chunk columns:
                  * Load input chunk (N_chunk x N) from DRAM to URAM_B
                  * For each row i in M_chunk:
                     - Locate identity vector for row i in URAM_A
                     - BF16_DOT_PRODUCT: identity_vector @ input_chunk → output_chunk
                     - Write output chunk to DRAM at transposed position
               b. Advance to next M_chunk

            Args:
                M: Number of rows in input matrix (becomes columns in output)
                N: Number of columns in input matrix (becomes rows in output, must be multiple of 64)
                input_dram_addr: DRAM address for input matrix (M x N) in bf16 format
                output_dram_addr: DRAM address for output matrix (N x M) in bf16 format
                program_dram_addr: DRAM address for instruction stream. If None, auto-allocated.
                params_dram_addr: DRAM address for identity matrix storage. If None, auto-allocated.

            Returns:
                handler: Callable that takes a torch.Tensor (M x N) and returns transposed
                         torch.Tensor (N x padded_M) in bf16 format. The output is padded
                         to M_aligned (next multiple of 64) for hardware alignment.
            """
            bytes_per_element = 2

            # Validate dimensions
            assert M > 0 and N > 0, "Matrix dimensions must be positive"
            assert N % UE_VECTOR_SIZE == 0, f"Matrix columns N={N} must be a multiple of {UE_VECTOR_SIZE}"

            # Calculate row count (padded to UE_VECTOR_SIZE)
            M_aligned = ((M - 1) // UE_VECTOR_SIZE + 1) * UE_VECTOR_SIZE
            output_aligned_elements = M_aligned * N

          # Auto-allocate params_dram_addr if not provided
            use_auto_mem_management_for_params = False
            if params_dram_addr is None:
                use_auto_mem_management_for_params = True
                params_dram_addr = self.get_params_dram_addr()

            identity_tensor = torch.eye(N).to(torch.bfloat16)

            # Write identity tensor to params DRAM (stored once during setup)
            identity_tensor_UE_SIZE = torch.eye(UE_VECTOR_SIZE).to(torch.bfloat16)
            self.dma_write(DMA_DEVICE_H2C, params_dram_addr, identity_tensor_UE_SIZE, UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element)

            if use_auto_mem_management_for_params:
                self.allocate_params_dram(UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element)

            # ===== Calculate URAM_B partitioning =====
            # Maximum N_chunk that fits in URAM_B with K columns
            max_N_chunk = ((URAM_NEAR_FULL_ELEMENTS // N) // UE_VECTOR_SIZE) * UE_VECTOR_SIZE # Lower to nearest multiple of 64
            max_N_chunk = min(max_N_chunk, M_aligned) # matrix is loaded into URAM_B with chunks of K * max_N_chunk elements
            assert max_N_chunk >= 1 and max_N_chunk <= M_aligned, f"max_N_chunk={max_N_chunk} must be greater than 0 and less than N={M_aligned}"

            max_M_chunk = min(N, URAM_HALF_ELEMENTS // N, URAM_HALF_ELEMENTS // max_N_chunk)
            assert max_M_chunk >= 1 and max_M_chunk <= N, f"max_M_chunk={max_M_chunk} must be greater than 0 and less than M={N}"

            # Start instruction capture
            self.start_capture()
            inst_id = 0

            input_uram_addr = URAM_START_ADDR
            input_chunk_dram_addr = params_dram_addr

            self.ue_memcpy_from_dram(input_chunk_dram_addr, UE_VECTOR_SIZE * UE_VECTOR_SIZE * bytes_per_element, 0, input_uram_addr, URAM_SECTION.URAM_A.value, inst_id)
            self.wait_queue()
            inst_id += 1

            # Process each weight chunk
            remaining_M_vectors = N
            output_chunk_dram_addr = output_dram_addr
            start_vector_index = 0
            while remaining_M_vectors > 0:
                current_max_M_chunk = min(max_M_chunk, remaining_M_vectors)
                current_M_chunk_bytes = current_max_M_chunk * N * bytes_per_element

                output_uram_addr = UE_VECTOR_SIZE # only 64x64 identity matrix is stored in URAM_A
                # Load max_M_chunk x K elements from DRAM to URAM_A

                remaining_N_chunks = M_aligned
                weight_chunk_dram_addr = input_dram_addr

                # weight matrix chunking is done here
                output_chunk_dram_offset_in_chunk_row = output_chunk_dram_addr
                while remaining_N_chunks > 0:
                    current_N_chunk = min(max_N_chunk, remaining_N_chunks)
                    current_N_chunk_bytes = current_N_chunk * N * bytes_per_element

                    # weight is always copied to URAM_B from start of URAM_B
                    self.ue_memcpy_from_dram(weight_chunk_dram_addr, current_N_chunk_bytes, 0, URAM_START_ADDR, URAM_SECTION.URAM_B.value, inst_id)
                    self.wait_queue()
                    inst_id += 1

                    # calculate one chunk of output for each input vector
                    for i in range(current_max_M_chunk):
                        ones_idx = identity_tensor[start_vector_index+i, :].reshape(-1, UE_VECTOR_SIZE).sum(axis=1).argmax(axis=0)
                        vector_idx = identity_tensor[start_vector_index+i, :].reshape(-1, UE_VECTOR_SIZE)[ones_idx, :].argmax(axis=0)

                        # BF16 dot product
                        self.start_queue(
                            0,  # broadcast_mode
                            0,  # clear_max_en not used
                            N // UE_VECTOR_SIZE,  # stride_z
                            LALU_MODE.BYPASS.value,  # lalu_mode
                            0,  # lalu_scalar
                            0,  # uram_bram
                            URAM_SECTION.URAM_A.value,  # uram_section
                            0,  # uram_dst_addr
                            0,  # dram_to_uram_cpy_start
                            output_uram_addr,  # uram_wb_addr
                            URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                            UE_MODE.BF16_DOT_PRODUCT,  # mode
                            0,  # data_type
                            input_uram_addr + vector_idx,  # uram_a_start_addr
                            URAM_START_ADDR + ones_idx,  # uram_b_start_addr
                            1,  # uram_length, 1 means one UE_VECTOR_SIZE elements
                            0,  # dma_start_addr not used
                            current_N_chunk * N,  # dma_length, need to provide full size of matrix
                            current_N_chunk,  # output_size
                            inst_id
                        )
                        inst_id += 1
                        self.wait_queue()

                        self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, output_uram_addr, output_chunk_dram_offset_in_chunk_row + i * M_aligned * bytes_per_element, current_N_chunk * bytes_per_element, inst_id)
                        self.wait_queue()
                        inst_id += 1

                    remaining_N_chunks -= current_N_chunk
                    output_chunk_dram_offset_in_chunk_row += current_N_chunk * bytes_per_element
                    weight_chunk_dram_addr += current_N_chunk_bytes # move weight chunk dram address to next chunk

                output_chunk_dram_addr += current_max_M_chunk * M_aligned * bytes_per_element
                remaining_M_vectors -= current_max_M_chunk
                input_chunk_dram_addr += current_M_chunk_bytes
                start_vector_index += current_max_M_chunk

            # Finish capture and write instruction stream to DRAM
            self.stop_capture()
            self.generate_instruction_halt()

            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
                self.allocate_program_dram(program_size_bytes)
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
                self.write_captured_instructions_to_dram(program_dram_addr)

            def handler(matrix):
                """
                Run transpose operation using the captured instruction stream.

                Supports DeviceTensor for DMA cache optimization - skips H2C DMA if
                input is already synced at the target address.

                Args:
                    matrix: Input matrix (M x N) - torch.Tensor or DeviceTensor

                Returns:
                    DeviceTensor with transposed matrix (N x padded_M), lazy C2H fetch
                """
                # Extract tensor data and check if DMA can be skipped
                if isinstance(matrix, DeviceTensor):
                    matrix_data = matrix._data  # Use _data to avoid triggering fetch
                    skip_dma = not matrix.needs_dma(input_dram_addr)
                    if skip_dma:
                        print(f"[DMA cache hit] Skipping input DMA to {hex(input_dram_addr)}")
                    else:
                        print(f"[DMA write] Writing input matrix to {hex(input_dram_addr)}")
                else:
                    matrix_data = matrix
                    skip_dma = False
                    print(f"Please consider using DeviceTensor for DMA cache optimization")

                # Validate input
                assert matrix_data.dtype == torch.bfloat16, "Input matrix must be in bf16 format"
                if matrix_data.shape != (M, N):
                    raise ValueError(f"Input matrix shape {matrix_data.shape} must match ({M}, {N})")

                # Write input matrix to DRAM (skip if DeviceTensor already synced)
                if skip_dma:
                    print(f"[DMA cache hit] Skipping input DMA to {hex(input_dram_addr)}")
                else:
                    if isinstance(matrix, DeviceTensor):
                        matrix.sync(input_dram_addr) # sync the data to the FPGA DRAM
                    else:
                        self.dma_write(DMA_DEVICE_H2C, input_dram_addr, matrix_data.flatten(), M * N * bytes_per_element)

                # Start executing from DRAM and wait for completion
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
                self.report_timing_and_instruction_count()
                # tranpose is elemental access for each element 1 ops per element
                print( f"flops: {self.report_flop_rate_gflops(output_aligned_elements):.3f} GFLOPS")

                if isinstance(matrix, DeviceTensor):
                    return DeviceTensor((N, M_aligned), ue=self, dram_addr=output_dram_addr)
                else:
                    return DeviceTensor((N, M_aligned), ue=self, dram_addr=output_dram_addr).data

            return handler

        def bf16_permute(self,
                         input_dram_addr: int,
                         output_dram_addr: int,
                         dim_0: int,
                         dim_1: int,
                         dim_2: int,
                         permute_indices: Optional[torch.Tensor] = None, # TODO: Think about using this
                         program_dram_addr: Optional[int] = None) -> Callable[[torch.Tensor], torch.Tensor]:
            """
            Run permute operation using the captured instruction stream.
            takes matrix dimension (dim_0, dim_1, dim_2) and return (dim_1, dim_0, dim_2)
            permute_indices is the indices to permute the matrix
            if permute_indices is not provided, it will be generated automatically
            """
            bytes_per_element = 2

            permute_a = torch.arange(0, dim_0*dim_1, dtype=torch.int32).reshape(dim_0, dim_1, 1)
            permute_a = permute_a.permute(1, 0, 2)
            permute_a = permute_a.flatten() # permute indices

            total_elements = dim_0 * dim_1 * dim_2

            # Start instruction capture
            self.start_capture()
            inst_id = 0

            output_chunk_dram_addr = output_dram_addr

            remaining_elements = total_elements
            near_full_elements_aligned = (URAM_NEAR_FULL_ELEMENTS // (UE_VECTOR_SIZE * dim_2)) * UE_VECTOR_SIZE * dim_2

            i = 0

            while remaining_elements > 0:
                current_elements = min(near_full_elements_aligned, remaining_elements)
                number_of_dim2_elements = current_elements // dim_2

                for j in range(number_of_dim2_elements):
                    self.ue_memcpy_from_dram(input_dram_addr + permute_a[i + j].item() * dim_2 * bytes_per_element, dim_2 * bytes_per_element, 0, URAM_START_ADDR + (j * dim_2) // UE_VECTOR_SIZE, URAM_SECTION.URAM_A.value, inst_id)
                    self.wait_queue()
                    inst_id += 1

                self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, URAM_START_ADDR, output_chunk_dram_addr, number_of_dim2_elements * dim_2 * bytes_per_element, inst_id)
                self.wait_queue()
                inst_id += 1

                remaining_elements -= current_elements
                output_chunk_dram_addr += number_of_dim2_elements * dim_2 * bytes_per_element
                i += number_of_dim2_elements

            # Finish capture and write instruction stream to DRAM
            self.stop_capture()
            self.generate_instruction_halt()

            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                print(f"Using program DRAM address: {program_dram_addr}")
                program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
                self.allocate_program_dram(program_size_bytes)
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
                self.write_captured_instructions_to_dram(program_dram_addr)

            def handler(matrix):
                """
                Run permute operation using the captured instruction stream.
                takes matrix dimension (dim_0, dim_1, dim_2) and return (dim_1, dim_0, dim_2)
                """
                # Extract tensor data and check if DMA can be skipped
                if isinstance(matrix, DeviceTensor):
                    matrix_data = matrix._data  # Use _data to avoid triggering fetch
                    skip_dma = not matrix.needs_dma(input_dram_addr)
                    if skip_dma:
                        print(f"[DMA cache hit] Skipping input DMA to {hex(input_dram_addr)}")
                    else:
                        print(f"[DMA write] Writing input matrix to {hex(input_dram_addr)}")
                else:
                    matrix_data = matrix
                    skip_dma = False
                    print(f"Please consider using DeviceTensor for DMA cache optimization")

                # Validate input
                assert matrix_data.dtype == torch.bfloat16, "Input matrix must be in bf16 format"

                # Write input matrix to DRAM (skip if DeviceTensor already synced)
                if skip_dma:
                    print(f"[DMA cache hit] Skipping input DMA to {hex(input_dram_addr)}")
                else:
                    if isinstance(matrix, DeviceTensor):
                        matrix.sync(input_dram_addr) # sync the data to the FPGA DRAM
                    else:
                        self.dma_write(DMA_DEVICE_H2C, input_dram_addr, matrix_data.flatten(), dim_0 * dim_1 * dim_2 * bytes_per_element)

                # Start executing from DRAM and wait for completion
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
                self.report_timing_and_instruction_count()
                # tranpose is elemental access for each element 1 ops per element
                print( f"flops: {self.report_flop_rate_gflops(total_elements):.3f} GFLOPS")

                if isinstance(matrix, DeviceTensor):
                    return DeviceTensor((dim_1, dim_0, dim_2), ue=self, dram_addr=output_dram_addr)
                else:
                    return DeviceTensor((dim_1, dim_0, dim_2), ue=self, dram_addr=output_dram_addr).data

            return handler

        def bf16_padding_zero(self,
                         input_dram_addr: int,
                         output_dram_addr: int,
                         M: int,
                         N: int,
                         program_dram_addr: Optional[int] = None,
                         params_dram_addr: Optional[int] = None) -> Callable[[torch.Tensor], torch.Tensor]:
            """
            Run padding zero operation using the captured instruction stream.
            takes matrix dimension (M, N) and return (M, N)
            padding zero is the padding zero to the matrix
            """
            bytes_per_element = 2

            # Start instruction capture
            self.start_capture()
            inst_id = 0

            N_padded = (N + UE_VECTOR_SIZE - 1) // UE_VECTOR_SIZE * UE_VECTOR_SIZE
            print(f"N_padded: {N_padded}")
          # Auto-allocate params_dram_addr if not provided
            if params_dram_addr is None:
                params_dram_addr = self.get_params_dram_addr()
                self.allocate_params_dram(N_padded * bytes_per_element)

            first_N_one = torch.zeros(N_padded, dtype=torch.bfloat16)
            first_N_one[:N] = 1.0

            # Write identity tensor to params DRAM (stored once during setup)
            self.dma_write(DMA_DEVICE_H2C, params_dram_addr, first_N_one.flatten(), N_padded * bytes_per_element)

            self.ue_memcpy_from_dram(params_dram_addr, N_padded * bytes_per_element, 0, URAM_START_ADDR, URAM_SECTION.URAM_B.value, inst_id)
            self.wait_queue()
            inst_id += 1

            for i in range(M):
                self.ue_memcpy_from_dram(input_dram_addr + i * N * bytes_per_element, N_padded * bytes_per_element, 0, URAM_START_ADDR + i * N_padded // UE_VECTOR_SIZE, URAM_SECTION.URAM_A.value, inst_id)
                self.wait_queue()
                inst_id += 1

                # Perform element-wise operation
                self.start_queue(
                    0,  # broadcast_mode (not used)
                    0,  # max_clear_en (not used)
                    1,  # stride_z
                    LALU_MODE.BYPASS.value,  # lalu_mode
                    0,  # scalar (not used)
                    0,  # uram_bram (URAM)
                    URAM_SECTION.URAM_A.value,  # uram_section (write to URAM_A)
                    0,  # uram_dst_addr
                    0,  # dram_to_uram_cpy_start
                    URAM_START_ADDR + i * N_padded // UE_VECTOR_SIZE,  # uram_wb_addr
                    URAM_WRITE_SRC.URAM_WRITE_BACK.value,  # uram_write_src
                    UE_MODE.ELTWISE_MUL,  # mode
                    0,  # data_type (not used)
                    URAM_START_ADDR + i * N_padded // UE_VECTOR_SIZE,  # uram_a_start_addr
                    URAM_START_ADDR,  # uram_b_start_addr
                    N_padded // UE_VECTOR_SIZE,  # uram_length
                    0,  # dma_start_addr (not used)
                    0,  # dma_length (not used)
                    0,  # output_size (not used)
                    inst_id  # inst_id
                )
                inst_id += 1
                self.wait_queue()

            self.ue_memcpy_to_dram(0, URAM_SECTION.URAM_A.value, URAM_START_ADDR, output_dram_addr, M * N_padded * bytes_per_element, inst_id)
            self.wait_queue()
            inst_id += 1
            # Finish capture and write instruction stream to DRAM
            self.stop_capture()
            self.generate_instruction_halt()

            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                print(f"Using program DRAM address: {program_dram_addr}")
                program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
                self.allocate_program_dram(program_size_bytes)
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
                self.write_captured_instructions_to_dram(program_dram_addr)

            def handler(matrix):
                """
                Run permute operation using the captured instruction stream.
                takes matrix dimension (dim_0, dim_1, dim_2) and return (dim_1, dim_0, dim_2)
                """
                # Extract tensor data and check if DMA can be skipped
                if isinstance(matrix, DeviceTensor):
                    matrix_data = matrix._data  # Use _data to avoid triggering fetch
                    skip_dma = not matrix.needs_dma(input_dram_addr)
                    if skip_dma:
                        print(f"[DMA cache hit] Skipping input DMA to {hex(input_dram_addr)}")
                    else:
                        print(f"[DMA write] Writing input matrix to {hex(input_dram_addr)}")
                else:
                    matrix_data = matrix
                    skip_dma = False
                    print(f"Please consider using DeviceTensor for DMA cache optimization")

                # Validate input
                assert matrix_data.dtype == torch.bfloat16, "Input matrix must be in bf16 format"

                # Write input matrix to DRAM (skip if DeviceTensor already synced)
                if skip_dma:
                    print(f"[DMA cache hit] Skipping input DMA to {hex(input_dram_addr)}")
                else:
                    if isinstance(matrix, DeviceTensor):
                        matrix.sync(input_dram_addr) # sync the data to the FPGA DRAM
                    else:
                        self.dma_write(DMA_DEVICE_H2C, input_dram_addr, matrix_data.flatten(), M * N * bytes_per_element)

                # Start executing from DRAM and wait for completion
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
                self.report_timing_and_instruction_count()
                # tranpose is elemental access for each element 1 ops per element
                print( f"flops: {self.report_flop_rate_gflops(M * N_padded):.3f} GFLOPS")

                if isinstance(matrix, DeviceTensor):
                    return DeviceTensor((M, N_padded), ue=self, dram_addr=output_dram_addr)
                else:
                    return DeviceTensor((M, N_padded), ue=self, dram_addr=output_dram_addr).data

            return handler

        def memcpy_dram_writeback_benchmark(self,
                         uram_src_addr: int,
                         output_dram_addr: int,
                         vector_size: int,
                         program_dram_addr: Optional[int] = None) -> Callable[[torch.Tensor], torch.Tensor]:
            """
            Run memcpy dram writeback operation using the captured instruction stream.
            """
            bytes_per_element = 2
            memcpy_length_bytes = vector_size * bytes_per_element
            print(f"total_size: {memcpy_length_bytes}")

            assert memcpy_length_bytes <= URAM_NEAR_FULL_SIZE, f"memcpy_length_bytes {memcpy_length_bytes} is greater than URAM_NEAR_FULL_SIZE {URAM_NEAR_FULL_SIZE}"
            # Start instruction capture
            self.start_capture()
            inst_id = 0

            self.ue_memcpy_to_dram(MEMCPY_TYPE.URAM.value, URAM_SECTION.URAM_A.value, uram_src_addr, output_dram_addr, memcpy_length_bytes, inst_id)
            self.wait_queue()
            inst_id += 1

            # Finish capture and write instruction stream to DRAM
            self.stop_capture()
            self.generate_instruction_halt()

            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                print(f"Using program DRAM address: {program_dram_addr}")
                program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
                self.allocate_program_dram(program_size_bytes)
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
                self.write_captured_instructions_to_dram(program_dram_addr)

            def handler():
                """
                Run memcpy dram writeback operation using the captured instruction stream.
                """
                # Start executing from DRAM and wait for completion
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
                self.report_timing_and_instruction_count()
                # tranpose is elemental access for each element 1 ops per element
                print( f"flops: {self.report_flop_rate_gflops(vector_size):.3f} GFLOPS")
                print(f"SRAM-DRAM transfer speed: {memcpy_length_bytes / self.report_latency_in_us():.3f} MB/s")
                return DeviceTensor((vector_size,), ue=self, dram_addr=output_dram_addr)

            return handler



        def memcpy_dram_readback_benchmark(self,
                         input_dram_addr: int,
                         uram_dst_addr: int,
                         vector_size: int,
                         program_dram_addr: Optional[int] = None) -> Callable[[torch.Tensor], torch.Tensor]:
            """
            Run memcpy dram readback operation using the captured instruction stream.
            """
            bytes_per_element = 2
            memcpy_length_bytes = vector_size * bytes_per_element
            print(f"total_size: {memcpy_length_bytes}")

            assert memcpy_length_bytes <= URAM_NEAR_FULL_SIZE, f"memcpy_length_bytes {memcpy_length_bytes} is greater than URAM_NEAR_FULL_SIZE {URAM_NEAR_FULL_SIZE}"
            # Start instruction capture
            self.start_capture()
            inst_id = 0

            self.ue_memcpy_from_dram(input_dram_addr, memcpy_length_bytes, MEMCPY_TYPE.URAM, uram_dst_addr, URAM_SECTION.URAM_A.value, inst_id)
            self.wait_queue()
            inst_id += 1

            # Finish capture and write instruction stream to DRAM
            self.stop_capture()
            self.generate_instruction_halt()

            if program_dram_addr is None:
                program_dram_addr = self.get_program_dram_addr()
                print(f"Using program DRAM address: {program_dram_addr}")
                program_size_bytes = self.write_captured_instructions_to_dram(program_dram_addr)
                self.allocate_program_dram(program_size_bytes)
            else:
                print(f"Using provided program DRAM address: {program_dram_addr}")
                self.write_captured_instructions_to_dram(program_dram_addr)

            def handler(matrix):
                """
                Run permute operation using the captured instruction stream.
                takes matrix dimension (dim_0, dim_1, dim_2) and return (dim_1, dim_0, dim_2)
                """
                # Extract tensor data and check if DMA can be skipped
                if isinstance(matrix, DeviceTensor):
                    matrix_data = matrix._data  # Use _data to avoid triggering fetch
                    skip_dma = not matrix.needs_dma(input_dram_addr)
                    if skip_dma:
                        print(f"[DMA cache hit] Skipping input DMA to {hex(input_dram_addr)}")
                    else:
                        print(f"[DMA write] Writing input matrix to {hex(input_dram_addr)}")
                else:
                    matrix_data = matrix
                    skip_dma = False
                    print(f"Please consider using DeviceTensor for DMA cache optimization")

                # Validate input
                assert matrix_data.dtype == torch.bfloat16, "Input matrix must be in bf16 format"

                # Write input matrix to DRAM (skip if DeviceTensor already synced)
                if skip_dma:
                    print(f"[DMA cache hit] Skipping input DMA to {hex(input_dram_addr)}")
                else:
                    if isinstance(matrix, DeviceTensor):
                        matrix.sync(input_dram_addr) # sync the data to the FPGA DRAM
                    else:
                        self.dma_write(DMA_DEVICE_H2C, input_dram_addr, matrix_data.flatten(), memcpy_length_bytes)

                # Start executing from DRAM and wait for completion
                self.start_execute_from_dram(program_dram_addr)
                self.wait_queue()
                self.report_timing_and_instruction_count()
                # tranpose is elemental access for each element 1 ops per element
                print( f"flops: {self.report_flop_rate_gflops(vector_size):.3f} GFLOPS")

                print(f"DRAM-SRAM transfer speed: {memcpy_length_bytes / self.report_latency_in_us():.3f} MB/s")

                return None

            return handler

# Single UnifiedEngine: base from core + Bf16OpsMixin (re-exported as UnifiedEngine)
_UnifiedEngineBase = UnifiedEngine

class UnifiedEngine(_UnifiedEngineBase, Bf16OpsMixin):
    """UnifiedEngine with core + BF16/high-level ops (unary_op_exp, layer_norm, matmat_mul, etc.)."""
    pass
