import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import unet_precompiled as up


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
    def test_phase_merge_matches_host(self):
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
        for i,value in enumerate(tensors[1:]):
            expected[i//2::2,i%2::2,64:] = value
        torch.testing.assert_close(output,expected)

    def test_memory_reuse_does_not_overwrite_live_skips(self):
        shapes = {name:(64,4,4) for name in ('input','a','b','c','d')}
        ops = [dict(output='a',inputs=['input']),dict(output='b',inputs=['a']),
               dict(output='c',inputs=['b']),dict(output='d',inputs=['a','c'])]
        layouts,_ = up.memory_plan(shapes,ops)
        self.assertEqual(layouts['d'].address,layouts['b'].address)
        self.assertNotEqual(layouts['a'].address,layouts['d'].address)
        self.assertNotEqual(layouts['c'].address,layouts['d'].address)

    def test_resolution_rejected_before_compilation(self):
        with self.assertRaises(ValueError):
            up.graph({},33,32)

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


if __name__ == '__main__':
    unittest.main()
