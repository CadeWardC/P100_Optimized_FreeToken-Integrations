"""Inspect a GGUF directory without downloading model weights (standard library only)."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import struct
import urllib.request


class Header:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def take(self, n):
        if n < 0 or n > len(self.data) - self.pos:
            raise ValueError("GGUF header exceeds the supplied byte limit")
        result = self.data[self.pos:self.pos + n]
        self.pos += n
        return result

    def number(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def string(self):
        return self.take(self.number("Q")).decode("utf-8")

    def value(self, kind, depth=0):
        formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if kind in formats:
            return self.number(formats[kind])
        if kind == 8:
            return self.string()
        if kind == 9 and depth < 4:
            element, count = self.number("I"), self.number("Q")
            if count > len(self.data):
                raise ValueError("invalid GGUF array count")
            for _ in range(count):
                self.value(element, depth + 1)
            return {"array_type": element, "count": count}
        raise ValueError(f"unsupported GGUF metadata type {kind}")


def inspect(data):
    header = Header(data)
    if header.take(4) != b"GGUF" or header.number("I") != 3:
        raise ValueError("expected a little-endian GGUF v3 header")
    count, kv_count = header.number("Q"), header.number("Q")
    if count > len(data) or kv_count > len(data):
        raise ValueError("invalid GGUF directory counts")
    metadata = {}
    for _ in range(kv_count):
        name = header.string()
        value = header.value(header.number("I"))
        if not name.startswith("tokenizer."):
            metadata[name] = value
    types = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 22: "IQ2_S", 30: "BF16"}
    tensors = []
    for _ in range(count):
        name, dims = header.string(), header.number("I")
        if not 1 <= dims <= 4:
            raise ValueError("invalid GGUF tensor dimension count")
        shape = [header.number("Q") for _ in range(dims)]
        kind, offset = header.number("I"), header.number("Q")
        tensors.append({"name": name, "shape": shape, "type": types.get(kind, str(kind)), "offset": offset})
    experts = [t for t in tensors if any(s in t["name"] for s in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_gate_up_exps.weight", "ffn_down_exps.weight"))]
    return {"metadata": metadata, "tensor_count": count, "header_bytes": header.pos,
            "tensor_types": dict(collections.Counter(t["type"] for t in tensors)),
            "expert_types": dict(collections.Counter(t["type"] for t in experts)), "experts": experts,
            "header_sha256": hashlib.sha256(data[:header.pos]).hexdigest(),
            "validation": "header only; full file hash and pretrained outputs not verified"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url")
    source.add_argument("--file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-mib", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.max_mib <= 64:
        parser.error("--max-mib must be between 1 and 64")
    limit = args.max_mib * 1024 * 1024
    if args.url:
        request = urllib.request.Request(args.url, headers={"Range": f"bytes=0-{limit - 1}"})
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206 or not response.headers.get("Content-Range", "").startswith("bytes 0-"):
                raise ValueError("server did not honor the bounded range request")
            data = response.read(limit)
    else:
        with args.file.open("rb") as handle:
            data = handle.read(limit)
    report = inspect(data)
    report["source"] = args.url or str(args.file.resolve())
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("tensor_count", "header_bytes", "expert_types", "validation")}))


if __name__ == "__main__":
    main()
