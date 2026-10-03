# syntax=docker/dockerfile:1.7
ARG EXL3_SOURCE_IMAGE=ghcr.io/tpurtell/glm-5.3-flash-exl3-4bpw-2x-rtx@sha256:48e254d94f58137c8707e6044cde4528c6af3fdd9702726b9b362e9b0e0b4629
ARG VLLM_BASE_IMAGE=vllm/vllm-openai@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8
FROM --platform=linux/amd64 ${EXL3_SOURCE_IMAGE} AS exl3_source
FROM ${VLLM_BASE_IMAGE}
ARG CUTE_DSL_ARCH=sm_120a
ARG B12X_VOCAB=0
ENV CUTE_DSL_ARCH=${CUTE_DSL_ARCH} QWEN38_B12X_VOCAB=${B12X_VOCAB}
ARG B12X_COMMIT=c76a40ee684cb3ef7d2c223d56a9b9cff25a3a1e
RUN B12X_COMMIT=${B12X_COMMIT} python3 - <<'PY'
import os, tarfile, urllib.request
from pathlib import Path
commit = os.environ['B12X_COMMIT']
urllib.request.urlretrieve(f'https://github.com/tpurtell/sparkinfer-glmrt/archive/{commit}.tar.gz', '/tmp/b12x.tar.gz')
with tarfile.open('/tmp/b12x.tar.gz') as archive:
    archive.extractall('/opt', filter='data')
Path(f'/opt/sparkinfer-glmrt-{commit}').rename('/opt/b12x')
Path('/tmp/b12x.tar.gz').unlink()
PY
RUN python3 -m pip install --no-deps -e /opt/b12x
COPY --from=exl3_source /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/exl3.py
COPY patches/port-exl3-qwen38.py /tmp/port-exl3-qwen38.py
COPY patches/b12x_qsa_attention.py /usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/b12x_qsa_attention.py
COPY patches/port-qsa-fp8.py /tmp/port-qsa-fp8.py
COPY patches/port-nvidia-mtp-fp8.py /tmp/port-nvidia-mtp-fp8.py
COPY patches/qwen_host_embedding.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/qwen_host_embedding.py
COPY patches/port-host-embedding.py /tmp/port-host-embedding.py
COPY patches/port-exl3-ple-fp8.py /tmp/port-exl3-ple-fp8.py
COPY patches/port-nvidia-ple-fp8.py /tmp/port-nvidia-ple-fp8.py
COPY patches/port-qwen-tool-constraints.py /tmp/port-qwen-tool-constraints.py
COPY patches/qwen_vocab_projection.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/qwen_vocab_projection.py
COPY patches/port-vocab-projection.py /tmp/port-vocab-projection.py
COPY patches/qwen_nvfp4_moe.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/qwen_nvfp4_moe.py
COPY patches/port-nvfp4-moe.py /tmp/port-nvfp4-moe.py
RUN python3 /tmp/port-exl3-qwen38.py /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-qsa-fp8.py /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-nvidia-mtp-fp8.py /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-host-embedding.py /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-nvidia-ple-fp8.py /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-exl3-ple-fp8.py /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-qwen-tool-constraints.py /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-vocab-projection.py /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/port-nvfp4-moe.py /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 -c 'from vllm.model_executor.layers.quantization import get_quantization_config; assert get_quantization_config("exl3").__name__ == "Exl3Config"'
COPY patches/port-ple-mmap.py patches/ple-mmap-pr54129.patch patches/ple-mmap-base-hashes.json /tmp/ple-mmap/
RUN python3 /tmp/ple-mmap/port-ple-mmap.py /usr/local/lib/python3.12/dist-packages/vllm
LABEL io.tpurtell.ple-mmap.pr="54129" \
      io.tpurtell.ple-mmap.commit="50a061f792f36364f5f95a93eee21f1e9d77f65e"
ENV VLLM_EXL3_TRELLIS_MIN_M=1 \
    VLLM_EXL3_PREFILL_TRELLIS=1 \
    VLLM_EXL3_PREFILL_CAPACITY=2048 \
    VLLM_PLE_MMAP=1 \
    VLLM_PLE_MMAP_SERIAL=128 \
    VLLM_PLE_MMAP_READAHEAD=2048 \
    VLLM_PLE_CPU_OFFLOAD=1 \
    VLLM_PLE_OFFLOAD_READY_TIMEOUT=1800
LABEL org.opencontainers.image.source="https://github.com/tpurtell/sm12x-exl3-qwen3.8-flash-next" \
      io.tpurtell.b12x.commit="${B12X_COMMIT}"

# Narrow upstream structured-output corrections; numerical kernels are unchanged.
COPY patches/port-xgrammar-termination.py patches/port-structured-output-reasoning.py /tmp/structured-output/
RUN python3 /tmp/structured-output/port-xgrammar-termination.py /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/structured-output/port-structured-output-reasoning.py /usr/local/lib/python3.12/dist-packages/vllm
COPY scripts/test-xgrammar-termination.py scripts/test-structured-output-reasoning.py /opt/qwen38-tests/
RUN python3 /opt/qwen38-tests/test-xgrammar-termination.py \
 && python3 /opt/qwen38-tests/test-structured-output-reasoning.py
LABEL io.tpurtell.structured-output.reasoning-fix="c6e19b3be243" \
      io.tpurtell.structured-output.termination-fix="vllm-pr-52805"

# Resident GPU workers own metadata only; the CPU subprocess owns the table.
COPY patches/port-ple-resident-loading.py /tmp/port-ple-resident-loading.py
RUN python3 /tmp/port-ple-resident-loading.py /usr/local/lib/python3.12/dist-packages/vllm

# Opt-in SSD prefetch and decode fairness. Numerical model code is unchanged.
COPY patches/port-ple-prefetch.py patches/port-ple-scheduling.py /tmp/ple-interference/
RUN python3 /tmp/ple-interference/port-ple-prefetch.py /usr/local/lib/python3.12/dist-packages/vllm \
 && python3 /tmp/ple-interference/port-ple-scheduling.py /usr/local/lib/python3.12/dist-packages/vllm
