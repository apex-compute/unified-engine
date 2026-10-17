import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parent))
import unet_precompiled as up
import unet_run_from_bin as runner


class MemoryEngine:
    """Byte-level DMA model for the transpose/skip merge (no arithmetic)."""
    def __init__(self):
        self.dram = bytearray(1 << 20)
        self.sram = bytearray(1 << 20)

    def accelerator_memory_to_sram(self,source,sram,unused,*,memcpy_length_bytes):
        self.sram[sram:sram+memcpy_length_bytes] = self.dram[source:source+memcpy_length_bytes]

    def sram_to_accelerator_memory(self,sram,destination,unused,*,memcpy_length_bytes,
                                    stride_bytes_per_chunk=0,stride_jump_bytes=0):
        chunk = stride_bytes_per_chunk or memcpy_length_bytes
        jump = stride_jump_bytes or chunk
        for start in range(0,memcpy_length_bytes,chunk):
            dst = destination+(start//chunk)*jump
            self.dram[dst:dst+chunk] = self.sram[sram+start:sram+start+chunk]


class WholeGraphTests(unittest.TestCase):
    def test_merge_only_copies_skip_half(self):
        engine = MemoryEngine()
        shapes = [(64,4,6)]+[(64,2,3)]*4
        sources = []
        tensors = []
        cursor = 0
        for i,shape in enumerate(shapes):
            layout = up.shared._layout_for_conv(shape)
            layout.address = cursor
            value = (torch.arange(layout.size_bytes//2,dtype=torch.int32)+i*1000).to(torch.int16)
            raw = value.numpy().tobytes()
            engine.dram[cursor:cursor+len(raw)] = raw
            tensors.append(value.reshape(shape[1],shape[2],shape[0]))
            sources.append(layout)
            cursor += layout.size_bytes
        dst = up.shared._layout_for_conv((128,4,6))
        dst.address = cursor
        up.emit_merge(engine,sources,dst)
        output = torch.frombuffer(engine.dram,dtype=torch.int16,count=dst.size_bytes//2,
                                  offset=dst.address).reshape(4,6,128)
        expected = torch.empty(4,6,128,dtype=torch.int16)
        expected[...,:64] = tensors[0]
        expected[...,64:] = 0
        torch.testing.assert_close(output,expected)

    def test_phase_conv_results_scatter_directly_into_merge(self):
        engine = MemoryEngine()
        dst = up.shared._layout_for_conv((128,4,6))
        dst.address = 0x80000
        expected = torch.zeros(4,6,128,dtype=torch.int16)
        for phase in range(4):
            values = (torch.arange(2*3*64,dtype=torch.int32)+phase*1000).to(torch.int16)
            raw = values.numpy().tobytes()
            start = up.shared._WB_SRAM_ADDRESS
            engine.sram[start:start+len(raw)] = raw
            up._scatter_phase_tile(engine,dst,(0,0,2,3),0,64,
                                   phase,64*2,0x60000)
            expected[phase//2::2,phase%2::2,64:] = values.reshape(2,3,64)
        output = torch.frombuffer(engine.dram,dtype=torch.int16,
            count=dst.size_bytes//2,offset=dst.address).reshape(4,6,128)
        torch.testing.assert_close(output,expected)

    def test_phase_oc32_uses_safe_contiguous_pixel_writes(self):
        engine = MemoryEngine()
        dst = up.shared._layout_for_conv((128,2,4))
        dst.address = 0x80000
        values = torch.arange(2*32,dtype=torch.int16)
        raw = values.numpy().tobytes()
        start = up.shared._WB_SRAM_ADDRESS
        engine.sram[start:start+len(raw)] = raw
        up._scatter_phase_tile(engine,dst,(0,0,1,2),32,32,1,128,0x60000)
        output = torch.frombuffer(engine.dram,dtype=torch.int16,
            count=dst.size_bytes//2,offset=dst.address).reshape(2,4,128)
        expected = torch.zeros_like(output)
        expected[0,1::2,96:128] = values.reshape(1,2,32)
        torch.testing.assert_close(output,expected)

    def test_memory_reuse_does_not_overwrite_live_skips(self):
        shapes = {name:(64,4,4) for name in ('input','a','b','c','d')}
        ops = [dict(op='conv',output='a',inputs=['input']),
               dict(op='conv',output='b',inputs=['a']),
               dict(op='conv',output='c',inputs=['b']),
               dict(op='conv',output='d',inputs=['a','c'])]
        layouts,_ = up.memory_plan(shapes,ops)
        self.assertEqual(layouts['d'].address,layouts['b'].address)
        self.assertNotEqual(layouts['a'].address,layouts['d'].address)
        self.assertNotEqual(layouts['c'].address,layouts['d'].address)

    def test_resolution_rejected_before_compilation(self):
        with self.assertRaises(ValueError):
            up.graph({},33,32)
        with self.assertRaises(ValueError):
            up.compile_hardware({},16,16,weight_reuse_pixels=0)

    def test_runtime_one_upload_kick_read(self):
        engine = mock.Mock()
        engine.dma_write.side_effect = lambda device,addr,data,size:size
        engine.dma_read.side_effect = lambda device,addr,data,size:(data.zero_(),size)[1]
        engine.is_queue_busy.return_value = False
        engine.read_reg32.return_value = up.udc.INT_CAUSE_HALT
        engine.read_latency_cycles.return_value = 123
        hardware = dict(resolution=[16,16],model_base=up.shared.MODEL_BASE,
                        model_image=torch.zeros(64,dtype=torch.uint8),program_address=up.shared.MODEL_BASE,
                        tensors={'outc':dict(address=up.shared.TENSOR_BASE,size_bytes=16*16*128,
                                             physical_channels=64)})
        backend = up.WholeGraphBackend(engine,hardware)
        output = backend.execute(torch.zeros(3,16,16))
        self.assertEqual(tuple(output.shape),(2,16,16))
        self.assertEqual(engine.dma_write.call_count,2)  # model + input
        engine.start_execute_from_dram.assert_called_once()
        engine.dma_read.assert_called_once()
        self.assertEqual((backend.kicks,backend.cycles),(1,123))

    def test_rk_runner_uses_reported_clock_and_single_bin_xdma_path(self):
        engine = mock.Mock()
        engine.dma_write.side_effect = lambda device,addr,data,size:size
        engine.dma_read.side_effect = lambda device,addr,data,size:(data.zero_(),size)[1]
        engine.is_queue_busy.return_value = False
        engine.read_reg32.return_value = up.udc.INT_CAUSE_HALT
        engine.read_latency_cycles.return_value = 123
        hardware = dict(resolution=[16,16],model_base=up.shared.MODEL_BASE,
                        model_image=torch.zeros(64,dtype=torch.uint8),
                        program_address=up.shared.MODEL_BASE,
                        tensors={'outc':dict(address=up.shared.TENSOR_BASE,
                            size_bytes=16*16*128,physical_channels=64)})
        payload = dict(full_graph=True,hardware=hardware,checkpoint_sha256='fixture')
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory)/'input.png'
            output = Path(directory)/'mask.png'
            Image.new('RGB',(16,16)).save(image)
            argv = ['unet_run_from_bin.py','--device','rk','--dev','xdma7',
                    '--backend','hardware','--resolution','16x16',
                    '--image',str(image),'--output',str(output)]
            with mock.patch.object(sys,'argv',argv), \
                 mock.patch.object(runner,'load_artifact',return_value=(payload,{})), \
                 mock.patch.object(up.udc,'set_dma_device') as select_device, \
                 mock.patch.object(up.udc,'configure_clock_from_hardware',return_value=3.25), \
                 mock.patch.object(up.udc,'configured_hardware_info',
                     return_value=types.SimpleNamespace(axi_data_width_bits=256)), \
                 mock.patch.object(up.udc,'UnifiedEngine',return_value=engine) as factory, \
                 contextlib.redirect_stdout(stdout):
                runner.main()
            self.assertTrue(output.is_file())
        select_device.assert_called_once_with('xdma7')
        factory.assert_called_once_with(clock_period_ns=3.25,
            conv_geometry_mode=up.udc.CONV_GEOMETRY_QUEUE_CONFIG)
        engine.software_reset.assert_called_once_with(run_dram_self_test=False)
        engine.start_execute_from_dram.assert_called_once_with(hardware['program_address'])
        self.assertEqual(engine.dma_write.call_count,2)
        engine.dma_read.assert_called_once()
        result = json.loads(stdout.getvalue().split('TEST_RESULT:',1)[1])
        self.assertEqual(result['program_kicks'],1)
        self.assertAlmostEqual(result['fpga_execution_s_sum'],123*3.25e-9)


if __name__ == '__main__':
    unittest.main()
