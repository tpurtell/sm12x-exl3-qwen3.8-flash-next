# SSD prefetch and decode fairness

The original mmap implementation abandons its entire explicit prefetch pass
when a gather exceeds `VLLM_PLE_MMAP_READAHEAD`. Prompt batches can therefore
delay every decoding request sharing their GPU iteration. These opt-in ports
change I/O scheduling, not embedding values, model weights, or numerical kernels.

## Build and enable

The main Dockerfile includes both ports. To layer them onto the qualified RTX
image without rebuilding CUDA kernels:

```sh
docker build -f Dockerfile.ple-fix \
  --build-arg BASE_IMAGE=ghcr.io/tpurtell/rtx6k-exl3-qwen3.8-flash-next@sha256:0f3fdb9e1073446ca756db7541bbcfe943a7e5c43aa97e85cbcba982d82b7516 \
  -t local/qwen38-ple-fix:trial .
```

Keep the baseline launch arguments unchanged for comparisons, including
`--max-num-batched-tokens 256`, and explicitly set:

```sh
VLLM_PLE_MMAP=1
VLLM_PLE_MMAP_READAHEAD=2048
VLLM_PLE_MMAP_PAGE_PREFETCH=1
VLLM_PLE_MMAP_RANDOM=1
VLLM_PLE_MMAP_DECODE_STEPS=3
```

All three new options default off. Page prefetch rounds requested row spans to
actual OS pages, merges overlapping/adjacent page spans, and hints bounded
worker-local windows before consuming them. Large batches continue through
successive windows rather than skipping all hints. Exact mmap row gathering
remains authoritative if a hint fails. Random advice is applied to the actual
memory mapping to suppress unnecessary automatic sequential readahead.

Decode fairness reuses the scheduler's existing pre-allocation prompt-deferral
branches. The configured number of eligible decode turns is followed by a
prefill opportunity. Without eligible decoding, prompt processing proceeds
normally. This trades some prompt completion time for decoding responsiveness;
it does not guarantee a maximum wall-clock token gap.
Enabled fairness requires data-parallel size one; other configurations fail
explicitly rather than mixing independently throttled scheduling policies.

These ports do **not** implement asynchronous next-batch GPU staging or a
disk-readiness scheduler. The captured CUDA graph still owns one staging buffer;
background writes to that buffer would be unsafe. Worker page hints overlap
within a gather; they do not hide the entire gather behind GPU computation.

## Verification and rollback

See [the QA plan](../tests/ws1-ple-prefetch-test-plan.md) and run the paired
synthetic benchmark. Record actual server usage separately from SSE chunks.
Compare decode-only throughput, mixed-workload stream gaps, and long-prompt
TTFT; an improvement in one must not conceal a regression in another.

Setting the three new options to `0` and restarting restores the original paths
within the patched image. Retain the original immutable image and complete
launch configuration for rollback. Do not flush the system page cache on a
shared host; process restarts alone are not cold-cache experiments.
