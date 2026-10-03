#!/usr/bin/env python3
"""Opt-in bounded decode turns between PLE prefills; not async PLE loading.

Usage: python port-ple-scheduling.py /path/to/vllm
Only the exact inspected scheduler is supported. Default behavior is unchanged.
"""
import hashlib
from pathlib import Path
import sys

BASE_SHA256 = "c710f49e41e974e5b7b8f1cd2f5fb9722523f4da0165252e16351199ebd03124"

HELPERS = '''
def _ple_decode_steps(environ):
    if environ.get("VLLM_PLE_MMAP", "0") != "1":
        return 0
    raw = environ.get("VLLM_PLE_MMAP_DECODE_STEPS", "0")
    try:
        value = int(raw)
    except (ValueError, TypeError) as exc:
        raise ValueError("VLLM_PLE_MMAP_DECODE_STEPS must be an integer in 0..32") from exc
    if not 0 <= value <= 32:
        raise ValueError("VLLM_PLE_MMAP_DECODE_STEPS must be an integer in 0..32")
    return value


def _ple_validate_parallel(limit, parallel_config):
    if limit and parallel_config.data_parallel_size != 1:
        raise ValueError("VLLM_PLE_MMAP_DECODE_STEPS requires data_parallel_size=1")


def _ple_defer_prefills(scheduler, can_schedule):
    limit = scheduler._ple_decode_step_limit
    if not limit:
        return False
    # Match running-request eligibility before counting a decode opportunity.
    has_decode = any(
        not request.is_prefill_chunk
        and scheduler.current_step >= request.next_decode_eligible_step
        and request.num_tokens_with_spec + request.num_output_placeholders
            - request.num_computed_tokens > 0
        and scheduler.max_model_len - request.num_computed_tokens
            - scheduler.num_sampled_tokens_per_step > 0
        and not (
            request.num_output_placeholders > 0
            and request.num_computed_tokens + 2 - request.num_output_placeholders
            >= request.num_prompt_tokens + request.max_tokens
        )
        for request in scheduler.running
    )
    if not has_decode:
        scheduler._ple_deferred_steps = 0
        return False
    if not can_schedule:
        return False
    if scheduler._ple_deferred_steps >= limit:
        scheduler._ple_deferred_steps = 0
        return False
    scheduler._ple_deferred_steps += 1
    return True


'''

INIT_ANCHOR = "        self.vllm_config = vllm_config\n"
DEFER_ANCHOR = '''        defer_prefills = (
            throttle_prefills and not self.prefill_capacity_bound
        ) and any(not r.is_prefill_chunk for r in self.running)
'''


def patched_source(data):
    actual = hashlib.sha256(data).hexdigest()
    if actual != BASE_SHA256:
        raise RuntimeError(f"PLE scheduling base mismatch: {actual} != {BASE_SHA256}")
    source = data.decode("utf-8")
    edits = [
        ("import itertools\n", "import itertools\nimport os\n"),
        ("class Scheduler(SchedulerInterface):\n", HELPERS + "class Scheduler(SchedulerInterface):\n"),
        (INIT_ANCHOR, INIT_ANCHOR + "        self._ple_decode_step_limit = _ple_decode_steps(os.environ)\n        _ple_validate_parallel(self._ple_decode_step_limit, vllm_config.parallel_config)\n        self._ple_deferred_steps = 0\n"),
        (DEFER_ANCHOR, DEFER_ANCHOR + '''        # PLE fairness is independent of the DP capacity-bound escape hatch.
        # Defer before KV allocation and token accounting, never in the worker.
        ple_defer = _ple_defer_prefills(
            self, not defer_prefills and token_budget > 0
            and input_budget > draft_slots
        )
        defer_prefills = defer_prefills or ple_defer
'''),
    ]
    for old, new in edits:
        if source.count(old) != 1:
            raise RuntimeError(f"PLE scheduling anchor mismatch: {old!r}")
        source = source.replace(old, new, 1)
    compile(source, "scheduler.py", "exec")
    return source


def main():
    target = Path(sys.argv[1]) / "v1/core/sched/scheduler.py"
    output = patched_source(target.read_bytes())
    target.write_bytes(output.encode("utf-8"))
    print("Applied opt-in PLE decode fairness (VLLM_PLE_MMAP_DECODE_STEPS=0..32)")


if __name__ == "__main__":
    main()
