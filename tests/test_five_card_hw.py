import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests import five_card_hw as hw


class FiveCardHardwareChecks(unittest.TestCase):
    def test_distinct_pci_functions(self):
        cards = [f"xdma{i}" for i in range(5, 10)]
        with tempfile.TemporaryDirectory() as tmp:
            sysfs = Path(tmp)
            for dev in cards:
                (sysfs / f"{dev}_user/device").mkdir(parents=True)
            self.assertEqual(hw.reserved_cards(",".join(cards), sysfs), cards)
            for invalid in (cards[:4], cards[:4] + cards[:1], cards[:4] + ["cpu"]):
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    hw.reserved_cards(",".join(invalid), sysfs)
            alias = sysfs / f"{cards[-1]}_user/device"
            alias.rmdir()
            alias.symlink_to(sysfs / f"{cards[0]}_user/device")
            with self.assertRaises(ValueError):
                hw.reserved_cards(",".join(cards), sysfs)

    def test_worker_must_generate_on_the_reserved_hardware(self):
        cards = [f"xdma{i}" for i in range(5, 10)]
        resident = {"devices": cards, "engines": 10}
        metrics = {"generated_tokens": 1, "resident": resident, "image_tokens": 64}
        job = SimpleNamespace(model="q38_27b", metrics=metrics, text="Red.", image="image.png")
        hw.check_reply(job, cards, r"\bred\b")
        for changes in ({"generated_tokens": 0}, {"stopped": True}, {"image_tokens": 0},
                        {"resident": {**resident, "devices": cards[::-1]}},
                        {"resident": {**resident, "engines": 5}}):
            with self.subTest(changes=changes), self.assertRaises(AssertionError):
                hw.check_reply(SimpleNamespace(**{**vars(job), "metrics": {**metrics, **changes}}), cards, "red")
        with self.assertRaises(AssertionError):
            hw.check_reply(job, cards, "blue")

    def test_build_uses_checkout_and_preserves_cached_decoder(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ):
            cache = Path(tmp) / "cache"
            shards = cache / "tp10"
            work = Path(tmp) / "run"
            for k in range(10):
                shard = shards / f"card_{k:02d}"
                shard.mkdir(parents=True)
                (shard / "layer_00.pt").touch()
                (shard / "decoder.bin").write_bytes(b"cached")
            (shards / "plan.json").write_text('{"P":10}')
            (cache / "weights").mkdir()
            hw.prepare("gptoss20b", cache, work)
            built = work / "models/gptoss20b/tp10/card_00"
            self.assertTrue((shards / "card_00/layer_00.pt").samefile(built / "layer_00.pt"))
            self.assertFalse((built / "decoder.bin").exists())
            (built / "decoder.bin").write_bytes(b"new")
            self.assertEqual((shards / "card_00/decoder.bin").read_bytes(), b"cached")
            self.assertTrue((work / "models/q35_9b/qwen3.5_2b_test.py").is_file())
            self.assertTrue((work / "user_dma_core.py").samefile(hw.ROOT / "user_dma_core.py"))
