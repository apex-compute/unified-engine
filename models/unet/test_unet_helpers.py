"""Hardware-independent tests for phase ordering and quantization."""
import sys
from pathlib import Path
import unittest
from unittest import mock
import torch
import torch.nn.functional as F

sys.path.insert(0,str(Path(__file__).resolve().parent))
from unet_common import phase_weights, interleave, quantize, Backend, udc

class UNetTests(unittest.TestCase):
    def test_large_tile_loads_one_bias_vector(self):
        # Exercise the real loop emitter and BRAM capacity check without DMA.
        # The failing U-Net tile has 40960 results, but only 64 bias values.
        for gather in (False, True):
            with self.subTest(gather=gather):
                engine = mock.Mock(spec=udc.UnifiedEngine)
                engine.capture_count = 0
                engine.get_program_dram_addr.return_value = 0x100000
                engine.alloc_isa_reg.side_effect = [1, 2]
                engine.loop_end.return_value = 16
                engine.accelerator_memory_to_bias_sram.side_effect = (
                    lambda *args: udc.UnifiedEngine.accelerator_memory_to_bias_sram(engine, *args))
                udc.UnifiedEngine._capture_conv2d_tile_loop(
                    engine, n_tiles=2, act_base=0x200000, act_bytes=760*128,
                    weights_dram_addr=0x400000, out_base=0x800000,
                    out_bytes=40960*2, kernel_w=3, kernel_h=3, ct=1,
                    oc_count=64, out_w=40, out_h=16, w_pad=42,
                    stride_s=1, dilation=1, data_type=udc.TYPE.IF8,
                    scale_dram_addr=0x600000, scale_count=64 if gather else 576,
                    bias_dram_addr=0x700000, results=40960,
                    lalu_mode=udc.LALU_MODE.BYPASS, lalu_a=0, lalu_b=0,
                    gather=gather, c_in=3 if gather else 64)
                engine.accelerator_memory_to_bias_sram.assert_called_once_with(0x700000,64)
                self.assertEqual(engine.ue_memcpy_from_dram.call_args.args[1],128)

    def test_transpose_phase_mapping_and_bias(self):
        torch.manual_seed(12)
        x = torch.randint(-2,3,(1,3,5,7)).float()
        weight = torch.randint(-2,3,(3,4,2,2)).float()
        bias = torch.arange(4).float()
        ref = F.conv_transpose2d(x,weight,bias,stride=2)[0]
        actual = interleave([F.conv2d(x,w,bias)[0] for w in phase_weights(weight)])
        torch.testing.assert_close(actual,ref,rtol=0,atol=0)

    def test_zero_and_signed_channel_quantization(self):
        weight = torch.tensor([0.,-2.,3.])[:,None,None,None].expand(3,65,3,3).clone()
        codes,scale = quantize(weight)
        self.assertTrue(torch.isfinite(scale).all())
        self.assertTrue((scale>0).all())
        self.assertTrue((codes[0]==0).all())
        self.assertTrue((codes[1]<0).all())
        restored = codes.float()*scale.float()[:,None,None,None]
        torch.testing.assert_close(restored,weight,atol=.02,rtol=0)

    def test_hardware_signed_scales_and_arena_lifetime(self):
        class Engine:
            last_conv_cycles = 23
            def run_conv2d_layer(self,x,w,**kwargs):
                self.kwargs=kwargs
                return x.clone()
            def reset_params_dram_addr(self): pass
            def reset_tensor_dram_addr(self): pass
            def reset_program_dram_addr(self): pass
        weight=torch.ones(64,3,3,3)
        codes,scales=quantize(weight)
        layer=dict(weight=weight,bias=torch.zeros(64),codes=codes,scales=scales,relu=True,pad=1)
        engine=Engine(); backend=Backend('hardware',engine)
        backend.conv(torch.ones(3,4,4),layer)
        self.assertTrue(engine.kwargs['gather'])
        self.assertEqual(tuple(engine.kwargs['block_scales'].shape),(64,1))
        self.assertTrue((engine.kwargs['block_scales']<0).all())
        self.assertEqual((backend.cycles,backend.kicks),(23,1))

if __name__ == '__main__': unittest.main()
