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

RUN uv sync --extra natten --extra fastapi

ENV PATH="/app/.venv/bin:$PATH"


# Setup and run on server
# hf download Lightricks/LTX-2.5 \
#     diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
#     text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
#     vae/ltx-2.5-video-vae-bf16.safetensors \
#     vae/ltx-2.5-audio-vae-bf16.safetensors \
#     loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors \
#     latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
#     --local-dir models/ltx-2.5

# uv run python -m ltx_pipelines.app \
#     --transformer-path       models/ltx-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
#     --text-encoder-path      models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
#     --video-vae-path         models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors \
#     --audio-vae-path         models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
#     --distilled-lora         models/ltx-2.5/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors 0.8 \
#     --spatial-upsampler-path models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
#     --offload cpu --prompt "" --output-path "output.mp4"

