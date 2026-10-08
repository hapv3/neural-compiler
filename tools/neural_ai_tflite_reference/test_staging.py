import struct
import unittest

from analyze_staging import replay_staging


def dma(kind, source, destination, length, sr=6, dims=()):
    data = bytearray(64)
    struct.pack_into("<HH", data, 0, kind, len(data))
    struct.pack_into("<HHI", data, 16, sr, 0, source)
    struct.pack_into("<HHI", data, 24, 6, 0, destination)
    struct.pack_into("<I", data, 32, length)
    for index, value in enumerate(dims):
        struct.pack_into("<I", data, 36 + index * 4, value)
    return bytes(data)


class StagingReplayTest(unittest.TestCase):
    def test_constant_seed_and_forward_copy_alias(self):
        commands = [dma(24, 0, 0, 4, sr=1), dma(3, 0, 4, 4, dims=(4, 4, 3))]
        actual, transfers = replay_staging(commands, bytes(range(16)), b"ABCD", 0, 2)
        self.assertEqual(actual, b"ABCD" * 4)
        self.assertTrue(transfers[-1]["bounding_ranges_overlap"])

    def test_3d_loop_order(self):
        command = dma(4, 0, 16, 2, dims=(2, 4, 2, 4, 8, 2))
        actual, _ = replay_staging([command], bytes(range(32)), b"", 0, 1)
        for index, address in enumerate((16, 20, 24, 28)):
            self.assertEqual(actual[address:address + 2], bytes((index * 2, index * 2 + 1)))

    def test_rejects_unsupported_compute_or_wide_local_copy(self):
        with self.assertRaisesRegex(ValueError, "Unsupported command"):
            replay_staging([dma(9, 0, 32, 4)], bytes(128), b"", 0, 1)
        with self.assertRaisesRegex(ValueError, "Unsupported local copy"):
            replay_staging([dma(2, 0, 64, 64)], bytes(128), b"", 0, 1)

    def test_rejects_out_of_bounds(self):
        with self.assertRaisesRegex(ValueError, "exceeds memory"):
            replay_staging([dma(2, 0, 31, 4)], bytes(32), b"", 0, 1)


if __name__ == "__main__":
    unittest.main()
