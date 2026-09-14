"""Streaming I/O tests use ordinary temporary files, never device handles."""

import contextlib
import errno
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dpdfnet8khz_engine import StreamingEngine, UnifiedEngine


class StreamingEngineIOTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = str(Path(self.directory.name) / "dma")
        Path(self.path).write_bytes(bytes(range(128)))
        # Skip hardware discovery and model allocation in the generic engine.
        with patch.object(UnifiedEngine, "__init__", return_value=None):
            self.engine = StreamingEngine()
        self.addCleanup(self.engine.close)
        # Every test forbids all vector syscall variants, including fallback.
        self.vector_calls = []
        for name in ("preadv", "pwritev", "readv", "writev"):
            patcher = patch(f"dpdfnet8khz_engine.os.{name}",
                            side_effect=AssertionError("vector I/O is forbidden"))
            mocked = patcher.start()
            self.vector_calls.append(mocked)
            self.addCleanup(patcher.stop)
            self.addCleanup(mocked.assert_not_called)

    def test_reuses_separate_read_and_write_descriptors_and_exact_offsets(self):
        source = torch.tensor([1.0, -2.0, 0.25], dtype=torch.bfloat16)
        target = torch.zeros_like(source)
        real_open = os.open
        with patch("dpdfnet8khz_engine.os.open", wraps=real_open) as opened:
            for offset in (16, 48):
                self.assertEqual(self.engine.dma_write(self.path, offset, source, 6), 6)
                self.assertEqual(self.engine.dma_read(self.path, offset, target, 6), 6)
                self.assertTrue(torch.equal(source, target))
            self.assertEqual(opened.call_count, 2)
        self.assertEqual(len(self.engine._dma_fds), 2)
        for fd in self.engine._dma_fds.values():
            self.assertEqual(os.lseek(fd, 0, os.SEEK_CUR), 54)
        contents = Path(self.path).read_bytes()
        self.assertEqual(contents[:16], bytes(range(16)))
        self.assertEqual(contents[22:48], bytes(range(22, 48)))

    def test_scalar_read_copies_into_offset_tensor_and_write_shares_storage(self):
        storage = torch.full((10,), 0x5A, dtype=torch.uint8)
        target = storage[2:8].view(torch.bfloat16)
        real_read, real_write = os.read, os.write

        def read(fd, size):
            self.assertEqual(os.lseek(fd, 0, os.SEEK_CUR), 32)
            self.assertEqual(size, 6)
            data = real_read(fd, size)
            self.assertIsInstance(data, bytes)
            return data

        def write(fd, data):
            self.assertEqual(os.lseek(fd, 0, os.SEEK_CUR), 64)
            self.assertEqual(torch.frombuffer(data, dtype=torch.uint8).data_ptr(), target.data_ptr())
            return real_write(fd, data)

        with patch("dpdfnet8khz_engine.os.read", side_effect=read), \
                patch("dpdfnet8khz_engine.os.write", side_effect=write):
            self.assertEqual(self.engine.dma_read(self.path, 32, target, 6), 6)
            self.assertEqual(self.engine.dma_write(self.path, 64, target, 6), 6)
        self.assertEqual(storage.tolist(), [0x5A, 0x5A, 32, 33, 34, 35, 36, 37, 0x5A, 0x5A])
        self.assertEqual(Path(self.path).read_bytes()[64:70], bytes(range(32, 38)))
        scalar = torch.zeros((), dtype=torch.uint8)
        self.assertEqual(self.engine.dma_read(self.path, 32, scalar, 1), 1)
        self.assertEqual(scalar.item(), 32)

    def test_short_read_preserves_remaining_bytes_and_reports_count(self):
        target = torch.full((8,), 0xA5, dtype=torch.uint8)
        self.assertEqual(self.engine.dma_read(self.path, 125, target, 8), 3)
        self.assertEqual(target.tolist(), [125, 126, 127, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.engine.dma_read(self.path, 128, target, 8), -1)
        self.assertEqual(target.tolist(), [125, 126, 127, 0xA5, 0xA5, 0xA5, 0xA5, 0xA5])

    def test_short_write_is_reported_without_replaying_the_transfer(self):
        with patch("dpdfnet8khz_engine.os.write", return_value=3) as write:
            self.assertEqual(self.engine.dma_write(self.path, 16, b"abcdef", 6), 3)
            write.assert_called_once()

    def test_noncontiguous_tensor_and_integer_keep_shared_copy_semantics(self):
        target = torch.zeros((2, 3), dtype=torch.uint8).T
        self.assertEqual(self.engine.dma_read(self.path, 16, target, 6), 6)
        self.assertEqual(target.tolist(), [[16, 17], [18, 19], [20, 21]])
        self.assertEqual(self.engine.dma_write(self.path, 32, 0x1234, 2), 2)
        self.assertEqual(Path(self.path).read_bytes()[32:34], b"\x34\x12")
        self.assertEqual(self.engine.dma_write(self.path, 40, -1, 4), 4)
        self.assertEqual(Path(self.path).read_bytes()[40:44], b"\xff" * 4)
        self.assertEqual(len(self.engine._dma_fds), 2)

    def test_only_established_seek_read_write_calls_are_used(self):
        target = bytearray(4)
        real_seek, real_read, real_write = os.lseek, os.read, os.write
        with patch("dpdfnet8khz_engine.os.lseek", wraps=real_seek) as seek, \
                patch("dpdfnet8khz_engine.os.read", wraps=real_read) as read, \
                patch("dpdfnet8khz_engine.os.write", wraps=real_write) as write:
            for offset in (8, 16):
                self.assertEqual(self.engine.dma_read(self.path, offset, target, 4), 4)
                self.assertEqual(target, bytes(range(offset, offset + 4)))
                self.assertEqual(self.engine.dma_write(self.path, offset + 64, target, 4), 4)
            self.assertEqual(seek.call_count, 4)
            self.assertEqual(read.call_count, 2)
            self.assertEqual(write.call_count, 2)
        for forbidden in self.vector_calls:
            forbidden.assert_not_called()
        contents = Path(self.path).read_bytes()
        self.assertEqual(contents[72:76], bytes(range(8, 12)))
        self.assertEqual(contents[80:84], bytes(range(16, 20)))

    def test_real_io_error_closes_failed_descriptor_without_retry(self):
        self.assertEqual(self.engine.dma_read(self.path, 0, bytearray(4), 4), 4)
        fd = self.engine._dma_fds[(self.path, "read")]
        for code in (errno.EIO, errno.EINVAL):
            with self.subTest(errno=code), \
                    patch("dpdfnet8khz_engine.os.read", side_effect=OSError(code, "failure")) as read, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(self.engine.dma_read(self.path, 0, bytearray(4), 4), -1)
                read.assert_called_once()
            self.assertFalse(self.engine._dma_fds)
            with self.assertRaises(OSError):
                os.fstat(fd)
        self.assertEqual(self.engine.dma_read(self.path, 0, bytearray(4), 4), 4)

    def test_sizes_and_mutable_buffers_match_shared_scalar_io(self):
        self.assertEqual(self.engine.dma_write(self.path, 16, b"ab", 4), 2)
        target = torch.zeros(2, dtype=torch.uint8)
        self.assertEqual(self.engine.dma_read(self.path, 32, target, 4), 4)
        self.assertEqual(target.tolist(), [32, 33])
        target = bytearray(2)
        self.assertEqual(self.engine.dma_read(self.path, 32, target, 4), 4)
        self.assertEqual(target, bytes(range(32, 36)))
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.engine.dma_write(self.path, 0, b"", 0), 0)
            self.assertEqual(self.engine.dma_read(self.path, 0, bytearray(), 0), -1)

    def test_closed_engine_does_not_open_device(self):
        self.engine.close()
        with patch("dpdfnet8khz_engine.os.open") as opened, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(self.engine.dma_write(self.path, 0, 7, 2), -1)
            self.assertEqual(self.engine.dma_read(self.path, 0, bytearray(4), 4), -1)
            opened.assert_not_called()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.engine._get_user_fd()

    def test_close_and_destructor_release_dma_and_inherited_user_handles_once(self):
        self.engine.user_device = self.path
        self.engine._get_user_fd()
        self.engine.dma_read(self.path, 0, bytearray(4), 4)
        self.engine.dma_write(self.path, 4, b"abcd", 4)
        fds = set(self.engine._dma_fds.values()) | {self.engine._user_fd}
        self.assertEqual(len(fds), 3)
        real_close = os.close
        with patch("dpdfnet8khz_engine.os.close", wraps=real_close) as closed:
            self.engine.__del__()
            self.engine.close()
            self.engine.__del__()
            self.assertEqual({call.args[0] for call in closed.call_args_list}, fds)
            self.assertEqual(closed.call_count, 3)
        self.assertIsNone(self.engine._user_fd)
        for fd in fds:
            with self.assertRaises(OSError):
                os.fstat(fd)

    def test_superclass_initialization_failure_cleans_up_early_io(self):
        fds = []

        def failing_init(engine):
            engine.user_device = self.path
            fds.append(engine._get_user_fd())
            self.assertEqual(engine.dma_read(self.path, 0, bytearray(4), 4), 4)
            fds.extend(engine._dma_fds.values())
            raise RuntimeError("initialization failed")

        with patch.object(UnifiedEngine, "__init__", failing_init):
            with self.assertRaisesRegex(RuntimeError, "initialization failed"):
                StreamingEngine()
        self.assertEqual(len(fds), 2)
        for fd in fds:
            with self.assertRaises(OSError):
                os.fstat(fd)


if __name__ == "__main__":
    unittest.main()
