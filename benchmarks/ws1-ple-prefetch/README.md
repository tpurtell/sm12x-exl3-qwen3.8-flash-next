# RTX PRO 6000 SSD PLE interference comparison

Measured October 3, 2026 UTC on one RTX PRO 6000 Blackwell Workstation Edition
(96 GB), Ryzen 5 9600X, approximately 29 GiB usable host RAM, and a Kingston
SNV3S1000G NVMe. Driver 595.91.07. This is not a DGX Spark result.

Baseline image: `sha256:0f3fdb9e1073446ca756db7541bbcfe943a7e5c43aa97e85cbcba982d82b7516`.
Candidate image: `sha256:bcb1237e8e7f95dbee4213107d58af2686b8f5ed076200b76372999aae13709a`.
Model: `nvidia/Qwen3.8-Flash-Next-NVFP4`, served revision
`2061e0b0c5d92bdf7c8fbd4241bbc2af239d7e2d`.
Both: TP1, MTP2, max model length 262144, max sequences 16, max batched tokens
256, FP8 KV, GPU memory fraction 0.8951546170584449, PLE mmap, 32 I/O workers,
read-ahead budget 2048, serial threshold 128. Candidate enables page-prefetch,
MADV_RANDOM, and three decode-only scheduling opportunities between prefill
opportunities. No numerical kernels or weights changed.

## Paired results

Every request completed with 768 output tokens and matching synthetic input
hashes. Mixed cases launch a long prompt after the ongoing stream begins.
The ongoing prompt is 249 tokens; the uncached long prompt is 82006 tokens.

| Metric | Baseline repeat 1 / 2 | Candidate repeat 1 / 2 |
|---|---:|---:|
| Mixed ongoing stream wall time | 18.31 / 17.66 s | 8.17 / 8.51 s |
| Mixed ongoing max SSE gap | 158 / 75 ms | 77 / 78 ms |
| Mixed ongoing p95 SSE gap | 75 / 68 ms | 72 / 72 ms |
| Long prompt first output | 21.73 / 21.10 s | 24.06 / 24.30 s |
| Long request total wall time | 27.54 / 26.26 s | 31.00 / 29.99 s |
| C1 short request wall time | 8.43 / 4.53 s | 5.57 / 5.15 s |
| C2 first stream wall time | 6.61 / 5.11 s | 5.96 / 5.28 s |
| C2 second stream wall time | 6.73 / 5.12 s | 6.27 / 5.20 s |

Observed mixed-stream completion improved 2.08–2.24x. This is primarily a
fairness result: p95 gaps did not consistently improve, and long-prompt first
output became 11–15% later. The second C1 candidate run was about 14% slower;
the second C2 pair was about 2–3% slower. Do not describe this as a universal
throughput improvement or as eliminating all disk stalls.

## Method and limits

Commands used the harness with `--run-id ws1-ple-oct03 --long-words 8192
--output-tokens 768 --timeout 300 --repeats 2`, against an isolated localhost
endpoint with no production traffic. There are two repetitions per case, not a
statistically powered study. Baseline ran before candidate. Each variant had
a fresh serving process; the OS page cache was not flushed or made equivalent.
The report does not claim cold SSD measurements or isolate the contribution of
each individual option. Prompt usage reports zero KV-cached input tokens for
these recorded requests; that says nothing about file-page cache residency.

SSE gaps are client chunk-arrival gaps, not exact per-token GPU latency. All
requests force the output length; this performance workload does not establish
semantic correctness. Raw artifacts contain synthetic hashes, timings and
usage, not user prompts or generated text.

Reproduce the comparison:

```sh
python scripts/compare-ple-interference.py benchmarks/ws1-ple-prefetch/baseline.json benchmarks/ws1-ple-prefetch/candidate.json
```

The [QA plan](../../tests/ws1-ple-prefetch-test-plan.md) lists correctness,
failure, rollback, and remaining coverage checks.

## Additional candidate checks

`candidate-long.json` records two simultaneous uncached 82006-token prompts,
each generating 768 tokens. First output arrived after 46.44 and 20.55 seconds;
total request duration was 52.47 and 31.52 seconds. Approximate output rates
after first chunk were 127 and 70 tokens/s respectively. Maximum SSE gaps were
22 and 79 ms. This is a candidate capacity check, not a paired speedup claim.

The candidate passed arithmetic JSON (`17 + 25 = 42`) and a forced
`lookup_weather(city="Paris")` tool-call smoke check. The complete repository
suite passed **244 tests** in the candidate image with GPU access; see
`regression-tests-gpu.txt`. An earlier run without GPU access had eight
CUDA-initialization failures, all resolved by running in the required environment.
