"""CPU policy tests; optional exact-source integration, not GPU validation."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).parents[1] / "patches/port-ple-scheduling.py"
SPEC = importlib.util.spec_from_file_location("ple_scheduling_port", PATH)
port = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(port)
namespace = {}
exec(port.HELPERS, namespace)
parse = namespace["_ple_decode_steps"]
defer = namespace["_ple_defer_prefills"]


def request(prefill=False, **kwargs):
    values = dict(is_prefill_chunk=prefill, next_decode_eligible_step=0,
                  num_output_placeholders=0, num_computed_tokens=10,
                  num_prompt_tokens=10, max_tokens=100, num_tokens_with_spec=11)
    values.update(kwargs)
    return SimpleNamespace(**values)


def scheduler(limit=3, running=None):
    return SimpleNamespace(_ple_decode_step_limit=limit, _ple_deferred_steps=0,
                           current_step=1, max_model_len=1024, num_sampled_tokens_per_step=1,
                           running=running if running is not None else [request()])


def test_opt_in_and_validation():
    assert parse({}) == 0
    assert parse({"VLLM_PLE_MMAP_DECODE_STEPS": "bad"}) == 0
    assert parse({"VLLM_PLE_MMAP": "1"}) == 0
    for n in (0, 3, 32):
        assert parse({"VLLM_PLE_MMAP": "1", "VLLM_PLE_MMAP_DECODE_STEPS": str(n)}) == n
    for value in ("-1", "33", "three", "3.5", ""):
        with pytest.raises(ValueError, match="0..32"):
            parse({"VLLM_PLE_MMAP": "1", "VLLM_PLE_MMAP_DECODE_STEPS": value})


@pytest.mark.parametrize("limit", [0, 1, 3, 32])
def test_bounded_cadence_and_prefill_progress(limit):
    state = scheduler(limit, [request(), request(prefill=True)])
    observed = [defer(state, True) for _ in range((limit + 1) * 4)]
    assert observed == ([True] * limit + [False]) * 4


def test_no_decode_resets_and_prefill_resumes_immediately():
    state = scheduler()
    assert defer(state, True)
    state.running = [request(prefill=True)]
    assert not defer(state, True)
    assert state._ple_deferred_steps == 0
    state.running.append(request())
    assert [defer(state, True) for _ in range(4)] == [True, True, True, False]


def test_paused_or_ineligible_steps_do_not_consume_budget():
    state = scheduler()
    for _ in range(10):
        assert not defer(state, False)
    assert state._ple_deferred_steps == 0
    for decode in (request(next_decode_eligible_step=3),
                   request(num_output_placeholders=1, num_computed_tokens=110)):
        state.running = [decode, request(prefill=True)]
        assert not defer(state, True)


def test_wrong_base_rejected():
    with pytest.raises(RuntimeError, match="base mismatch"):
        port.patched_source(b"unexpected source")


def test_no_token_work_or_model_space_cannot_defer_prefill():
    for decode in (request(num_tokens_with_spec=10), request(num_tokens_with_spec=9),
                   request(num_computed_tokens=1023, num_tokens_with_spec=1024)):
        state = scheduler(running=[decode, request(prefill=True)])
        assert not defer(state, True)
        assert state._ple_deferred_steps == 0


def test_parallel_gate():
    validate = namespace["_ple_validate_parallel"]
    validate(3, SimpleNamespace(data_parallel_size=1))
    validate(0, SimpleNamespace(data_parallel_size=2))
    with pytest.raises(ValueError, match="data_parallel_size=1"):
        validate(3, SimpleNamespace(data_parallel_size=2))


def test_exact_scheduler_allocation_predicates():
    """Evaluate the real allocation guards, including prefix-cache decode hits.

    Set VLLM_SCHEDULER_SOURCE to the unmodified pinned scheduler. The fixture
    remains outside this small port repository; CI can use its image source.
    """
    path = os.environ.get("VLLM_SCHEDULER_SOURCE")
    if not path:
        pytest.skip("Set VLLM_SCHEDULER_SOURCE for exact-source integration")
    source = port.patched_source(Path(path).read_bytes())
    tree = ast.parse(source)
    schedule = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "schedule")
    guards = [n for n in ast.walk(schedule) if isinstance(n, ast.If)
              and "defer_prefills" in ast.unparse(n.test)]
    running = next(n for n in guards if "request.is_prefill_chunk" in ast.unparse(n.test))
    waiting = next(n for n in guards if "num_computed_tokens" in ast.unparse(n.test))
    allocations = sorted(n.lineno for n in ast.walk(schedule) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute) and n.func.attr == "allocate_slots")
    assert running.lineno < allocations[0] < waiting.lineno < allocations[1]
    assert any(isinstance(n, ast.Continue) for n in ast.walk(running))
    assert any(isinstance(n, ast.Break) for n in ast.walk(waiting))
    for enabled in (False, True):
        for prefill in (False, True):
            req = request(prefill=prefill)
            req.num_tokens = 100
            context = dict(defer_prefills=enabled, request=req,
                           num_computed_tokens=20 if prefill else 99)
            for guard in (running, waiting):
                decision = eval(compile(ast.Expression(guard.test), "guard", "eval"), context)
                assert decision == (enabled and prefill)
    # Actual combined predicate preserves DP behavior and ignores capacity for PLE.
    state = scheduler()
    state.prefill_capacity_bound = True
    scope = dict(self=state, throttle_prefills=False, token_budget=256,
                 input_budget=256, draft_slots=2, _ple_defer_prefills=defer)
    # Extract the inserted assignments directly rather than reimplementing policy.
    assignments = [n for n in schedule.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id in ("defer_prefills", "ple_defer") for t in n.targets)]
    exec(compile(ast.Module(body=assignments, type_ignores=[]), "policy", "exec"), scope)
    assert scope["defer_prefills"] is True
    # Independently requested DP deferrals never consume/reset the PLE release.
    # Otherwise alternating external and PLE phases could starve all prefills.
    state.prefill_capacity_bound = False
    state._ple_deferred_steps = 3
    scope["throttle_prefills"] = True
    policy = compile(ast.Module(body=assignments, type_ignores=[]), "policy", "exec")
    exec(policy, scope)
    assert scope["defer_prefills"] is True
    assert state._ple_deferred_steps == 3
    scope["throttle_prefills"] = False
    exec(policy, scope)
    assert scope["defer_prefills"] is False
    assert state._ple_deferred_steps == 0
