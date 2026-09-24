"""Measure weight H2D/kernel overlap in an Nsight Systems SQLite export."""
import argparse
import bisect
from collections import defaultdict
import json
from pathlib import Path
import sqlite3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--staging-chunk', type=int, default=4093,
                        help='full weight staging chunk used by test-moe-offload')
    args = parser.parse_args()
    groups = defaultdict(list)
    intersections = []
    overlapping_copies = 0
    with sqlite3.connect(args.trace) as db:
        for start, end, device, context, stream in db.execute(
                'select start,end,deviceId,contextId,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start'):
            groups[(device, context, stream)].append((start, end))
        starts = {key: [s for s, _ in value] for key, value in groups.items()}
        copies = list(db.execute(
            'select start,end,deviceId,contextId,streamId from CUPTI_ACTIVITY_KIND_MEMCPY '
            'where copyKind=1 and bytes=? order by start', (args.staging_chunk,)))
        for start, end, device, context, stream in copies:
            overlapped = False
            for key, kernels in groups.items():
                if key[:2] != (device, context) or key[2] == stream:
                    continue
                index = max(0, bisect.bisect_right(starts[key], start) - 1)
                while index < len(kernels) and kernels[index][0] < end:
                    lo, hi = max(start, kernels[index][0]), min(end, kernels[index][1])
                    if lo < hi:
                        intersections.append((lo, hi))
                        overlapped = True
                    index += 1
            overlapping_copies += overlapped
    total = 0
    if intersections:
        intervals = sorted(intersections)
        lo, hi = intervals[0]
        for start, end in intervals[1:]:
            if start > hi:
                total += hi - lo
                lo, hi = start, end
            else:
                hi = max(hi, end)
        total += hi - lo
    result = {'trace': str(args.trace), 'staging_chunk_bytes': args.staging_chunk,
              'weight_h2d_chunks': len(copies), 'chunks_overlapping_kernels': overlapping_copies,
              'overlap_union_us': total / 1000,
              'scope': 'Full-size weight chunks on a different stream, same device and CUDA context; excludes tail chunks.'}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not overlapping_copies:
        raise SystemExit('No GPU transfer/compute overlap observed')


if __name__ == '__main__':
    main()
