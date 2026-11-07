"""Host-only checks for the conv router (UnifiedEngine.conv2d_auto strategy mode). No board."""
import unittest
from unittest.mock import MagicMock

import user_dma_core as core


def geom(k=3, s=1, d=1, cin=128, cout=128, H=32, W=32, pad=None, pad_h=None, kh=None):
    return core.ConvGeom(c_in=cin, in_h=H, in_w=W, kernel_h=k if kh is None else kh, kernel_w=k, stride_s=s,
                         pad=d * (k - 1) // 2 if pad is None else pad, pad_h=pad_h, dilation=d, oc_count=cout)


class ConvRouterTests(unittest.TestCase):
    def test_default_rule(self):
        pick = core.choose_conv_strategy
        one_d = dict(H=1, W=3000, kh=1, pad_h=0)
        self.assertEqual(pick(geom(cin=256, cout=256)), "wino")
        self.assertEqual(pick(geom(cin=64, cout=64)), "native")
        self.assertEqual(pick(geom(cin=512, cout=3)), "native")
        self.assertEqual(pick(geom(s=2, cin=256, cout=256)), "native")
        self.assertEqual(pick(geom(k=7, cin=768, cout=768, **one_d)), "wino")
        self.assertEqual(pick(geom(k=7, cin=192, cout=192, **one_d)), "native")
        self.assertEqual(pick(geom(k=14, s=14, pad=0, cin=3, cout=1152, H=224, W=224)), "naive")
        self.assertEqual(pick(geom(k=4, s=4, pad=0, cin=3, cout=192, H=384, W=384)), "native")

    def test_unsupported_reasons(self):
        why = core.conv_strategy_unsupported
        self.assertIn("stride 2", why(geom(s=2), "wino"))
        self.assertIn("1x1", why(geom(k=1), "wino"))
        self.assertIn("> 15", why(geom(k=16, s=16, pad=0, cin=3, cout=768, H=512, W=512), "native"))
        self.assertIn("bf16", why(geom(), "wino", core.TYPE.IF8))
        for dt in (core.TYPE.IF8, core.TYPE.IF4):
            self.assertIsNone(why(geom(), "native", dt))
            self.assertIsNone(why(geom(), "naive", dt))
            self.assertEqual(core.choose_conv_strategy(geom(cin=256, cout=256), dt), "native")
        self.assertIsNone(why(geom(), "naive"))

    def test_geometry(self):
        g = geom(k=7, H=15, W=20)
        self.assertEqual((g.Hout, g.Wout, g.wino["nph"], g.wino["npw"]), (15, 20, 3, 3))
        g = geom(k=3, d=9, cin=48, cout=48, H=1, W=1000, kh=1, pad_h=0)
        self.assertEqual((g.wino["dim"], g.wino["gs"]), (1, 9))

    def test_impossible_strategy_raises_before_upload(self):
        ue = MagicMock(spec=core.UnifiedEngine)
        with self.assertRaisesRegex(ValueError, "stride 2"):
            core.UnifiedEngine.conv2d_auto(
                ue, strategy="wino", weight=core.torch.zeros(64, 64, 3, 3), ACT_DRAM_ADDR=0, OUTPUT_DRAM_ADDR=0,
                c_in=64, in_h=8, in_w=8, kernel_h=3, kernel_w=3, stride_s=2, pad=1, oc_count=64)
        ue.dma_write.assert_not_called()

    def test_legacy_dispatch_unchanged(self):
        hit = []
        core.UnifiedEngine.conv2d_auto(
            MagicMock(spec=core.UnifiedEngine), path="winograd", winograd=lambda: hit.append(1), ACT_DRAM_ADDR=0,
            WEIGHTS_DRAM_ADDR=0, SCALE_DRAM_ADDR=0, OUTPUT_DRAM_ADDR=0, c_in=128, in_h=8, in_w=8, kernel_h=3,
            kernel_w=3, stride_s=1, pad=1, oc_count=128, data_type=core.TYPE.FLOAT16)
        self.assertEqual(hit, [1])


if __name__ == "__main__":
    unittest.main()
