import struct
import unittest

from inspect_model_header import inspect


def string(value):
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


class ModelHeaderTests(unittest.TestCase):
    def fixture(self):
        data = b"GGUF" + struct.pack("<IQQ", 3, 2, 1)
        data += string("general.architecture") + struct.pack("<I", 8) + string("gemma4")
        for name, shape in [("blk.0.ffn_gate_up_exps.weight", (32, 128, 9)),
                            ("blk.0.ffn_down_exps.weight", (64, 32, 9))]:
            data += string(name) + struct.pack("<IQQQIQ", 3, *shape, 2, 0)
        return data

    def test_fused_directory(self):
        result = inspect(self.fixture())
        self.assertEqual(result["expert_types"], {"Q4_0": 2})
        self.assertEqual(result["experts"][0]["shape"], [32, 128, 9])
        self.assertEqual(result["header_bytes"], len(self.fixture()))

    def test_truncation(self):
        data = self.fixture()
        for length in (0, 7, 23, 80, len(data) - 1):
            with self.subTest(length=length), self.assertRaises(ValueError):
                inspect(data[:length])

    def test_corrupt_counts(self):
        with self.assertRaises(ValueError):
            inspect(b"GGUF" + struct.pack("<IQQ", 3, 2**63, 0))


if __name__ == "__main__":
    unittest.main()
