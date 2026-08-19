uvx hf download Lightricks/LTX-2.5 \
    diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
    text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    vae/ltx-2.5-video-vae-bf16.safetensors \
    vae/ltx-2.5-audio-vae-bf16.safetensors \
    latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors \
    --local-dir models/ltx-2.5

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

uv sync --extra server --group kernels && uv pip install 'flash-attn-4==4.0.0b9'

uv run python -m ltx_pipelines.app \
    --transformer-path       models/ltx-2.5/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors \
    --text-encoder-path      models/ltx-2.5/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors \
    --video-vae-path         models/ltx-2.5/vae/ltx-2.5-video-vae-bf16.safetensors \
    --audio-vae-path         models/ltx-2.5/vae/ltx-2.5-audio-vae-bf16.safetensors \
    --spatial-upsampler-path models/ltx-2.5/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors \
    --distilled-lora         models/ltx-2.5/loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors \
    --compile mode=max-autotune fullgraph=true dynamic=true --diffvae-optimization blackwell_dsl \
    --quantization nvfp4-cast --max-batch-size 16 --prompt "" --output-path "output.mp4"
