FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:0.11.6 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    UV_COMPILE_BYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends git python3-pip python3-dev build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN uv venv /app/.venv

COPY . .

RUN uv sync --extra server --group kernels && uv pip install 'flash-attn-4==4.0.0b9'

ENV PATH="/app/.venv/bin:$PATH"

