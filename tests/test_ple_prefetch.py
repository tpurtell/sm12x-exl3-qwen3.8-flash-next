"""Run the actual patched table code on CPU; no vLLM/GPU import required."""
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import math
import mmap
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prefetch_port', ROOT / 'patches/port-ple-prefetch.py')
port = importlib.util.module_from_spec(spec)
spec.loader.exec_module(port)


def base_source():
    patch = (ROOT / 'patches/ple-mmap-pr54129.patch').read_text(encoding='utf-8')
    section = patch.split('+++ b/vllm/models/qwen3_8_flash_next/nvidia/ple_mmap.py\n', 1)[1]
    section = section.split('\n--- ', 1)[0]
    return ''.join(line[1:] + '\n' for line in section.splitlines() if line.startswith('+'))


@pytest.fixture
def runtime(monkeypatch):
    calls = []
    monkeypatch.setattr(os, 'posix_fadvise', lambda *args: calls.append(args), raising=False)
    monkeypatch.setattr(os, 'POSIX_FADV_WILLNEED', 3, raising=False)
    logger = SimpleNamespace(**{name: lambda *a, **k: None for name in
                              ('info', 'info_once', 'warning', 'warning_once')})
    namespace = dict(np=np, os=os, mmap=mmap, time=time, math=math,
                     ThreadPoolExecutor=ThreadPoolExecutor, logger=logger,
                     get_dtype_size=lambda _: 1, _HAS_POSIX_FADVISE=True,
                     _LOG_INTERVAL_S=60.)
    tree = ast.parse(port.transform(base_source()))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in ('_coalesce_runs', '_ple_page_runs', 'MmapPleTable')]
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)] + nodes,
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<patched PLE>', 'exec'), namespace)
    return namespace, calls


def test_strict_hash_matches_repository_backport():
    assert hashlib.sha256(base_source().encode()).hexdigest() in (ROOT / 'patches/port-ple-prefetch.py').read_text(encoding='utf-8')


@pytest.mark.parametrize('offsets,width,expected', [
    ([], 160, []), ([17, 177, 337], 160, [(0, 4096)]),
    ([4090], 160, [(0, 8192)]), ([1, 8193], 160, [(0, 4096), (8192, 4096)]),
    ([4096, 8192], 4096, [(4096, 8192)]),
])
def test_page_coverage(runtime, offsets, width, expected):
    ns, _ = runtime
    assert ns['_ple_page_runs'](offsets, width, 4096) == expected


@pytest.mark.parametrize('optimized', [False, True])
@pytest.mark.parametrize('serial', [0, 10000])
def test_over_budget_exact_rows_and_prefetch(runtime, monkeypatch, tmp_path, optimized, serial):
    ns, calls = runtime
    monkeypatch.setenv('VLLM_PLE_MMAP_PAGE_PREFETCH', str(int(optimized)))
    # Offset deliberately crosses page boundaries; data includes repeated IDs.
    data = np.random.default_rng(13).integers(0, 256, (1000, 160), dtype=np.uint8)
    path = tmp_path / 'table.bin'
    path.write_bytes(b'x' * 4079 + data.tobytes())
    table = ns['MmapPleTable']({0: (str(path), 4079, 1000)}, 1000, 160, None,
                               workers=2, chunk=128, model_path=str(tmp_path), readahead=4, serial=serial)
    ids = np.array([999, 0, 100, 700, 200, 500, 300, 100, 600])
    try:
        np.testing.assert_array_equal(table.gather(ids), data[ids])
        assert bool(calls) == optimized  # Old all-or-none cutoff abandons all hints.
        if optimized:
            assert len(calls) > table.readahead  # Rolling windows reach all ranges.
            assert all(offset % mmap.PAGESIZE == length % mmap.PAGESIZE == 0
                       for _, offset, length, _ in calls)
        assert table.gather(np.array([], dtype=np.int64)).shape == (0, 160)
        with pytest.raises(IndexError):
            table.gather(np.array([-1]))
    finally:
        table.close()


def test_failed_advice_preserves_values(runtime, monkeypatch, tmp_path):
    ns, _ = runtime
    monkeypatch.setenv('VLLM_PLE_MMAP_PAGE_PREFETCH', '1')
    def fail(*args):
        raise OSError('advice unavailable')
    monkeypatch.setattr(os, 'posix_fadvise', fail)
    path = tmp_path / 'table.bin'
    data = np.arange(640, dtype=np.uint8).reshape(4, 160)
    path.write_bytes(data.tobytes())
    table = ns['MmapPleTable']({0: (str(path), 0, 4)}, 4, 160, None,
                               workers=2, chunk=2, model_path=str(tmp_path), readahead=1)
    try:
        np.testing.assert_array_equal(table.gather(np.array([3, 0, 3])), data[[3, 0, 3]])
        assert table._shard_fds == [None]
    finally:
        table.close()


@pytest.mark.parametrize('enabled', [False, True])
def test_random_advice_targets_mapping(runtime, monkeypatch, enabled):
    ns, _ = runtime
    hints = []
    monkeypatch.setenv('VLLM_PLE_MMAP_RANDOM', str(int(enabled)))
    monkeypatch.setattr(mmap, 'MADV_RANDOM', 1, raising=False)
    monkeypatch.setattr(np, 'memmap', lambda *a, **k:
                        SimpleNamespace(_mmap=SimpleNamespace(madvise=hints.append)))
    table = ns['MmapPleTable']({0: ('unused', 0, 4)}, 4, 160, None,
                               workers=1, chunk=2, model_path='unused')
    try:
        assert hints == ([mmap.MADV_RANDOM] if enabled else [])
    finally:
        table.close()
