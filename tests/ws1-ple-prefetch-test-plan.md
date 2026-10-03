# WS1 PLE prefetch QA plan

## Scope and prerequisites

Validate bounded SSD prefetch behavior and concurrent generation on the WS1 RTX PRO 6000 runtime. The primary risk is that uncached prompt embedding gathers delay an existing decoding request. Run the runtime's focused Python tests before live testing. This plan requires Python 3.10+ for the standard-library benchmark harness, an authorized WS1 shell, the same model checkpoint and GPU for both variants, and permission to restart the serving process. No ARTEMIS device is involved: this is an HTTP inference runtime, not a mobile application.

## Executed validation, October 3, 2026

- Windows Python 3.14 and candidate Linux Python 3.12: all 24 focused tests passed,
  including exact pinned scheduler integration. Local fake HTTP harness tests,
  compilation, and paired-result validation passed.
- Candidate Linux image with RTX PRO 6000 GPU access: full suite **244 passed**.
  The earlier no-GPU run had eight CUDA-initialization failures, not a passing
  regression run.
- Live isolated baseline/candidate: C1, C2 and mixed, two repetitions each,
  identical prompt hashes and 768-token output lengths; all streams completed.
- Candidate: two concurrent uncached 82006-token prompts completed; arithmetic,
  JSON response, and forced tool invocation smoke checks passed.
- Saved managed deployment restarted on the tested image with all three options
  persisted. Normal serving endpoint returned HTTP 200; arithmetic JSON and
  forced tool-call checks passed again after deployment.
- [Full results and caveats](../benchmarks/ws1-ple-prefetch/README.md) include
  the prompt-latency tradeoff and slower second C1 result. OS page-cache state
  was not controlled; no universal or cold-cache speedup is claimed.
- Pending: hours-long soak, other GPUs/storage devices, real user task replay,
  cancellation/preemption under sustained saturation, and isolated attribution
  of performance to each individual flag. Data-parallel fairness is unsupported
  and explicitly rejected when enabled.

The local benchmark self-test needs no GPU, credentials, external network, or real model:

```sh
python scripts/benchmark-ple-interference.py --self-test
python scripts/benchmark-ple-interference.py --help
```

The server endpoint defaults to `http://127.0.0.1:8000/v1`, with served model alias `qwen3.8-flash-next-fast`. Run the harness on WS1 or through an explicitly established localhost tunnel. Set `OPENAI_API_KEY` in the environment if the endpoint requires authentication; never put a key in commands, reports or URLs. Confirm no unrelated requests are running before comparative tests. Baseline and candidate must have the same model revision, launch flags (including max batched tokens 256), concurrency limits, thinking mode, output length and cache preparation.

## How to use the benchmark

Build using [the runtime instructions](../docs/ssd-prefetch-fairness.md). Candidate
settings are `VLLM_PLE_MMAP_PAGE_PREFETCH=1`, `VLLM_PLE_MMAP_RANDOM=1`, and
`VLLM_PLE_MMAP_DECODE_STEPS=3`, with mmap enabled, read-ahead budget 2048,
32 workers, and max batched tokens 256. New settings default off. The fairness
option requires data-parallel size one. Retain the baseline image and launch
configuration before restarting.

Focused checks:

```sh
VLLM_SCHEDULER_SOURCE=/path/to/unmodified/scheduler.py python -m pytest tests/test_ple_prefetch.py tests/test_ple_scheduling.py -q
```

The source fixture must come from the pinned baseline image; the port verifies
its hash. Without that variable, the exact-source integration test is skipped
and must not be reported as passed.

1. Record the actual baseline runtime revision, container/image identifier, launch flags, system RAM, GPU model, SSD, filesystem, and model revision alongside results. Save only synthetic benchmark artifacts. Do not copy user prompts or responses.
2. Start the baseline runtime. Use a defined, repeatable cache preparation procedure and record it. A process restart resets process state but does **not** clear the OS page cache. Do not describe a restart alone as cold SSD cache. Avoid globally dropping caches on a shared machine.
3. Run the following benchmark with no other inference traffic:

   ```sh
   python scripts/benchmark-ple-interference.py --label baseline --run-id ws1-pair-1 --seed 20261002 --short-words 16 --long-words 4096 --output-tokens 512 --timeout 600 --mixed-delay 0.5 --repeats 3 --output benchmarks/ws1-ple-baseline.json
   ```

4. Stop the baseline service and start the candidate with otherwise identical configuration. Repeat the identical command with `--label candidate` and `--output benchmarks/ws1-ple-candidate.json`. Keep `--run-id`, seed, prompt sizes and all workload options identical so prompt hashes match. The request nonce differs between cases/repeats to reduce prefix-cache reuse within one run. The script does not forcibly bypass or flush server prefix caches.
5. Compare each matching case and repeat. Inspect request status, server usage prompt/completion tokens, first-chunk latency, p95 and maximum inter-chunk gaps, total request wall time, and raw chunk arrival times. `usage_completion_tokens_per_wall_s` includes prefill and is not pure decode throughput. Chunk count is **not** a token count. Each timestamp records arrival at the HTTP client, including transport buffering.
6. For `mixed`, confirm `valid: true`, that the long request begins before the decoding request ends, and that long-prompt prefill actually overlaps ongoing output in raw timestamps. If decode ends too early, increase `--output-tokens` or reduce `--mixed-delay`, then rerun both variants with the new settings. Valid overlap alone does not guarantee enough samples for a meaningful comparison.
7. Repeat matched comparisons with `--long-words 256`, `4096`, and a larger value within the actual model context budget (for example `8192`). These are synthetic record counts, not token counts; inspect `usage.prompt_tokens`. Increase request timeout explicitly if required, preserving it between variants. Record timeouts as failures, never discard them as slow outliers.
8. Restore the intended service configuration after measurement. Benchmark JSON is the only persistent harness output; remove only the specific generated files when cleanup is desired. The harness never changes the service or runtime settings.

## Test cases and expected results

| ID | Preconditions and steps | Observable expected result |
| --- | --- | --- |
| PLE-01 | Run `--self-test` locally. | Exit 0 and PASS output. Local fake HTTP exercises C1/C2/mixed, reasoning SSE arrivals, separate chunk/token counts, EOF errors, request deadline and deterministic synthetic inputs. |
| PLE-02 | Execute C1 baseline/candidate with `--cases c1`. | Both return complete SSE streams with non-null server usage and finish reason. Compare short-prompt decode gaps and throughput; flag material regression for investigation. |
| PLE-03 | Execute C2 with `--cases c2`. | Two independent requests start at a thread barrier, complete without errors, and show sustained overlapping generation. Compare each stream's gaps and usage counts. Do not use `n=2` as a substitute for independent requests. |
| PLE-04 | Execute mixed with `--cases mixed`. | Long synthetic request begins after the ongoing request's first text/reasoning chunk, while that request is still active. Candidate should reduce the ongoing stream's prompt-correlated pauses. Record both the gap reduction and any long-prompt TTFT cost; no numerical improvement is presumed. |
| PLE-05 | Compare small and large synthetic prompts under the same cache preparation. | No all-or-nothing prefetch abandonment merely because the number of ranges crosses the prior cutoff; verify through runtime counters/logs or focused tests. Candidate uses bounded work and completes every gather with correct data. |
| PLE-06 | Test duplicate/shared-page IDs and boundary IDs through runtime focused tests. | Prefetch grouping preserves requested embedding values and ordering; adjacent/shared pages do not cause redundant unbounded hint work. Out-of-range IDs retain expected runtime behavior. |
| PLE-07 | Test zero/disabled prefetch option through focused tests and, if applicable, a controlled live run. | Explicitly disabled prefetch keeps supported prior semantics; no unintended file I/O scheduling or changed embedding results. |
| PLE-08 | Send invalid model name to a local test instance or fake endpoint. | Non-200 response becomes a failed request and benchmark exit 1. No response/error body containing potentially sensitive text is saved. |
| PLE-09 | Use deliberately short deadline against a local fake slow server. | Socket is aborted, the request records an error and wall time, and the script does not report successful completion. |
| PLE-10 | Check identical seed/run ID baseline/candidate artifacts. | Corresponding prompt SHA-256 values match; raw prompts, generated text and API keys are absent from JSON. Different case/repeat nonce changes hashes within a run. |
| PLE-11 | Restart actual candidate runtime and run short model smoke inference. | Runtime imports and starts without errors; response schema and normal reasoning/content generation still work. Source-only tests cannot satisfy this case. |

## Edge cases and potential bugs

- Large prompts can exceed context capacity; the harness records the HTTP failure. Choose record counts based on actual token usage, not word/token assumptions.
- Early EOS is suppressed with the vLLM `ignore_eos`/`min_tokens` parameters to keep comparison length stable. Other OpenAI-compatible implementations may reject those extensions; this harness targets the stated vLLM runtime.
- Thinking is enabled by default and reasoning fields count as output chunks. Use `--no-thinking` only as a separately matched experiment.
- Missing usage is not substituted with chunk count. A completed stream without usage cannot establish token throughput; mark that comparison inconclusive.
- Cold/warm OS pages and prefix/KV cache state can materially change results. Alternate baseline/candidate order across independent matched pairs when practical and record cache preparation; do not claim controlled cold-cache measurements without evidence.
- A long prompt may finish prefill after ongoing decoding has already finished. Inspect overlap and timestamps before drawing conclusions. Consider longer fixed output lengths when this occurs.
- Read-ahead may increase disk contention or cache churn. Record disk read bandwidth, queue depth, CPU I/O wait, GPU utilization and PLE gather timing during each live case. Faster prefetch submission does not by itself establish faster inference.
- Keep service restarts separate from benchmark execution. Ctrl+C is not a server restart; the harness has no service-management capability. Completed cases are written after each case, while interrupted in-flight cases may not be present.

## Gaps and limitations

The harness measures chunk arrival latency rather than per-token latency and
does not control server or OS caches. This implementation supplies bounded
worker-side page prefetch and scheduler fairness, not asynchronous next-batch
GPU staging or a wall-clock latency guarantee. Cases PLE-05 through PLE-07 have
focused exact-row/page-range/fallback test coverage; long-term storage behavior
and isolated performance attribution remain untested. Additional gaps are
listed in the executed-validation section above.

## Validation evidence

- 2026-10-02, Windows local Python: `python scripts/benchmark-ple-interference.py --self-test` passed. An ephemeral loopback HTTP server verified C1/C2/mixed execution, 12 SSE chunks versus 24 reported tokens, nonzero timing metrics, truncated streams, deadline failure and deterministic prompts.
- Live WS1 paired baseline/candidate and additional long-context test: **passed**,
  with measured tradeoffs documented in the results.
- Candidate build, startup, 24 focused checks, 244 regression checks, arithmetic,
  structured JSON and tool-call smoke: **passed**.
- Manual soak, cancellation/saturation, alternative hardware and real-task replay:
  **proposed, not completed**.
