"""Fail-closed raw-memory provenance/layout and per-command PMU checks."""

import copy
import csv
import hashlib
import io
import struct
import unittest
import zlib

import numpy as np

from verify_afu_mean import audit_command_pmu, audit_compute_dumps, c32_tensor, scratch_offset


def sha(data):
    return hashlib.sha256(data).hexdigest()


class EvidenceTest(unittest.TestCase):
    def setUp(self):
        self.command = struct.pack('<HHIII', 14, 64, 16, 9, 0) + bytes(48)
        self.plan = dict(command=1, completion_command=1, type=14, size=64, flags=16, layer_id=9,
                         tile_id=0, command_sha256=sha(self.command))
        self.memory = bytes(512 * 1024)
        self.compressed = zlib.compress(self.memory, 1)
        self.entry = dict(self.plan, path='capture', compressed_bytes=len(self.compressed),
                          compressed_sha256=sha(self.compressed), tcdm_bytes=len(self.memory),
                          tcdm_sha256=sha(self.memory))

    def audit(self, entries, commands=None, data=None):
        return audit_compute_dumps(entries, [self.plan], commands or [self.command],
                                   lambda path: self.compressed if data is None else data, sha)

    def test_valid_capture_and_missing_duplicate_or_extra_entries(self):
        self.assertEqual(self.audit([self.entry]), {1: self.memory})
        for entries in ([], [self.entry, self.entry], [dict(self.entry, command=2)]):
            with self.subTest(entries=len(entries)), self.assertRaises(ValueError):
                self.audit(entries)

    def test_tampered_command_compressed_data_size_or_raw_hash(self):
        for field, value in (('completion_command', 2), ('compressed_bytes', 0), ('tcdm_bytes', 1),
                             ('compressed_sha256', 'bad'), ('tcdm_sha256', 'bad')):
            entry = copy.deepcopy(self.entry)
            entry[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit([entry])
        with self.assertRaises(ValueError):
            self.audit([self.entry], commands=[self.command[:-1] + b'1'])
        with self.assertRaises(ValueError):
            self.audit([self.entry], data=self.compressed + b'corrupted')

    def test_c32_decode_ignores_channel_padding_and_obeys_offset(self):
        for shape in ((1, 3, 5, 33), (1, 7, 65)):
            channels, spatial = shape[-1], int(np.prod(shape[1:-1]))
            logical = np.arange(spatial * channels, dtype=np.int64).astype(np.int8).reshape(shape)
            groups = (channels + 31) // 32
            blocked = np.full((groups, spatial, 32), -123, np.int8)
            for pixel in range(spatial):
                for channel in range(channels):
                    blocked[channel // 32, pixel, channel % 32] = logical.reshape(spatial, channels)[pixel, channel]
            memory = b'prefix' + blocked.tobytes() + b'tail'
            np.testing.assert_array_equal(c32_tensor(memory, 6, shape), logical)
            for offset in (-1, len(memory)):
                with self.assertRaises(ValueError):
                    c32_tensor(memory, offset, shape)
        for shape in ((2, 7, 3), (1, 0, 3), (1, 3)):
            with self.assertRaises(ValueError):
                c32_tensor(self.memory, 0, shape)

    def test_scratch_reference_rejects_binding_alias_and_misalignment(self):
        self.assertEqual(scratch_offset(struct.pack('<HHI', 6, 0, 64), 0), 64)
        for ref in ((4, 0, 64), (6, 1, 64), (6, 0, 65)):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                scratch_offset(struct.pack('<HHI', *ref), 0)

    def test_pmu_descriptor_sequence_and_cycle_accounting(self):
        fields = ('command', 'type', 'size', 'flags', 'layer_id', 'tile_id', 'cycle')
        good = dict(command=1, type=14, size=64, flags=16, layer_id=9, tile_id=0, cycle=123)

        def stream(rows):
            value = io.StringIO()
            writer = csv.DictWriter(value, fields)
            writer.writeheader(); writer.writerows(rows)
            value.seek(0)
            return value

        self.assertEqual(audit_command_pmu(stream([good]), [self.command], 123)['command_cycles'], 123)
        for rows in ([], [good, good], [dict(good, type=13)], [dict(good, command=2)],
                     [dict(good, cycle=124)], [dict(good, cycle=-1)]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                audit_command_pmu(stream(rows), [self.command], 123)


if __name__ == '__main__':
    unittest.main()
