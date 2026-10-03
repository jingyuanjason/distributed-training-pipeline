# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.0 AS uv
FROM python:3.12-slim-bookworm

# PyTorch's locked Linux wheels install CUDA/NCCL user-space libraries.
# The GPU driver is supplied by the Kubernetes node, not this image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH="/opt/venv/bin:$PATH" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Keep a compiler available for TorchInductor/Triton runtime compilation.
# NCCL IB transport needs the verbs library and hardware providers (including mlx5).
# Include RDMA diagnostics; host drivers and devices are supplied by the cluster.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates libgomp1 \
        libibverbs1 ibverbs-providers ibverbs-utils rdma-core \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app

# Cache third-party dependencies independently of changes to project source.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project

# Kubeflow supplies the launcher command and arguments.
ENTRYPOINT []
CMD []