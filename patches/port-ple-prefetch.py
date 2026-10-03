#!/usr/bin/env python3
"""Opt-in page-aware rolling PLE prefetch for the pinned mmap backport.

VLLM_PLE_MMAP_PAGE_PREFETCH=1 enables bounded worker-local prefetch.
VLLM_PLE_MMAP_RANDOM=1 independently disables speculative mmap readahead.
Both default to zero to allow an unchanged baseline and independent A/B runs.
"""
import hashlib
from pathlib import Path
import sys

HELPERS = '''
def _ple_page_runs(offsets, row_bytes, page_size):
    """Cover precisely touched pages, merging overlaps and adjacent pages."""
    runs = []
    for offset in offsets:
        start = int(offset) // page_size * page_size
        end = ((int(offset) + row_bytes + page_size - 1) // page_size) * page_size
        if runs and start <= runs[-1][0] + runs[-1][1]:
            previous, length = runs[-1]
            runs[-1] = (previous, max(previous + length, end) - previous)
        else:
            runs.append((start, end - start))
    return runs


'''


def transform(source):
    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError(f'PLE prefetch anchor mismatch: {old[:90]!r}')
        source = source.replace(old, new, 1)

    replace('import math\n', 'import math\nimport mmap\n')
    replace('class MmapPleTable:', HELPERS + 'class MmapPleTable:')
    replace('        self.serial = max(0, int(serial))\n', '''        self.serial = max(0, int(serial))
        self.page_prefetch = os.getenv("VLLM_PLE_MMAP_PAGE_PREFETCH", "0") == "1"
        self.random_access = os.getenv("VLLM_PLE_MMAP_RANDOM", "0") == "1"
        self.page_size = mmap.PAGESIZE
        # Each active worker issues at most this many hints before consuming
        # those rows. Total issued lookahead is bounded by max(workers, budget).
        self.prefetch_rows = max(1, self.readahead // self.workers)
''')
    replace('            self.rows_total += rows\n', '''            if self.random_access:
                mapping = self.mm[idx]._mmap
                if hasattr(mapping, "madvise") and hasattr(mmap, "MADV_RANDOM"):
                    try:
                        mapping.madvise(mmap.MADV_RANDOM)
                    except (OSError, ValueError) as exc:
                        logger.warning_once("PLE mmap: MADV_RANDOM failed: %s", str(exc))
            self.rows_total += rows
''')
    replace('        total_runs = 0\n', '''        if self.page_prefetch:
            # Diagnostics only: actual hints are issued in bounded worker
            # windows immediately before copying, never an all-or-none pass.
            return sum(
                len(_ple_page_runs(self.mm[int(shard[s])].offset
                                  + local[s:e] * self.row_bytes,
                                  self.row_bytes, self.page_size))
                for s, e in zip(seg_starts, seg_ends)
                if self.mm[int(shard[s])] is not None
                and self._shard_fds[int(shard[s])] is not None
            )
        total_runs = 0
''')
    replace('            out[a:b] = mm[local[a:b]]\n', '''            if self.page_prefetch and self.readahead > 0:
                for first in range(a, b, self.prefetch_rows):
                    last = min(first + self.prefetch_rows, b)
                    fd = self._shard_fds[si]
                    if fd is not None:
                        offsets = mm.offset + local[first:last] * self.row_bytes
                        for offset, length in _ple_page_runs(
                            offsets, self.row_bytes, self.page_size
                        ):
                            try:
                                os.posix_fadvise(fd, offset, length, os.POSIX_FADV_WILLNEED)
                            except OSError as exc:
                                logger.warning_once("PLE mmap: page prefetch failed: %s", str(exc))
                                for idx, shard_fd in enumerate(self._shard_fds):
                                    if shard_fd == fd:
                                        self._shard_fds[idx] = None
                                break
                    # Prefetch is advisory. This exact original data read
                    # remains the correctness path, even after hint failure.
                    out[first:last] = mm[local[first:last]]
            else:
                out[a:b] = mm[local[a:b]]
''')
    return source


def main():
    path = Path(sys.argv[1]) / 'models/qwen3_8_flash_next/nvidia/ple_mmap.py'
    # Normalize line endings only; Windows diagnostic copies use CRLF.
    source = path.read_text(encoding='utf-8')
    expected = 'd07075530463a40a843d635327ec41907bf953f519eb75889159799cf0257f38'
    actual = hashlib.sha256(source.encode()).hexdigest()
    if actual != expected:
        raise RuntimeError(f'PLE prefetch base mismatch: {actual} != {expected}')
    path.write_text(transform(source), encoding='utf-8')
    print('Applied opt-in bounded page prefetch and mmap MADV_RANDOM')


if __name__ == '__main__':
    main()
