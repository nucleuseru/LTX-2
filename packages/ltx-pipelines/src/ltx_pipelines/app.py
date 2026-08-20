"""FastAPI server defining an OpenAI-compatible API for video generation using LTX-2 pipeline."""

from __future__ import annotations

import argparse
import base64
import logging
import os
import random
import tempfile
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Literal
from contextlib import asynccontextmanager

import torch
import asyncio
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    add_generated_keyframes_arg,
    default_2_stage_distilled_arg_parser,
    resolve_cli_params,
)
from ltx_pipelines.utils.media_io import (
    encode_video,
    resolve_hdr_color_space,
    vae_dtype_for_hdr,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ltx_server")

JOBS: dict[str, dict[str, Any]] = {}
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    params = resolve_cli_params(distilled=True)
    parser = add_generated_keyframes_arg(
        default_2_stage_distilled_arg_parser(params=params, supports_auto_duration=True)
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host IP")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    return parser.parse_args()


GLOBAL_ARGS = parse_args()

pipeline = DistilledPipeline(
    model_paths=GLOBAL_ARGS.model_paths,
    spatial_upsampler_path=GLOBAL_ARGS.spatial_upsampler_path,
    loras=tuple(GLOBAL_ARGS.lora) if GLOBAL_ARGS.lora else (),
    quantization=GLOBAL_ARGS.quantization,
    compilation_config=GLOBAL_ARGS.compile,
    offload_mode=GLOBAL_ARGS.offload_mode,
    prompt_enhancer_gemma_root=GLOBAL_ARGS.prompt_enhancer_gemma_root,
    diffvae_optimization=GLOBAL_ARGS.diffvae_optimization,
)


# Models
class InputReference(BaseModel):
    image_url: str | None = None
    frame_idx: int = 0  # 0 = first frame, -1 = last frame


class CreateVideoRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    prompt: str
    input_references: list[InputReference] | None = None
    input_reference: InputReference | None = None
    model: str = "lightricks/ltx-2.5"
    seconds: str | int = "10"
    size: str = "1280x764"


class VideoError(BaseModel):
    code: str
    message: str


class VideoObject(BaseModel):
    id: str
    object: Literal["video"] = "video"
    created_at: int
    completed_at: int | None = None
    expires_at: int | None = None
    status: Literal["queued", "in_progress", "completed", "failed"]
    progress: int
    model: str
    prompt: str | None = None
    seconds: str
    size: str
    remixed_from_video_id: str | None = None
    error: VideoError | None = None


class DeleteVideoResponse(BaseModel):
    id: str
    deleted: bool = True
    object: Literal["video.deleted"] = "video.deleted"


class VideoListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[VideoObject]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


# Helpers
def prepare_images(
    refs: list[InputReference], num_frames: int
) -> list[ImageConditioningInput]:
    images = []
    for ref in refs:
        if not ref.image_url:
            continue

        if ref.image_url.startswith("data:image/"):
            data = base64.b64decode(ref.image_url.split(",", 1)[1])
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(data)
            tmp.close()
            path = tmp.name
        elif ref.image_url.startswith(("http://", "https://")):
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            with urllib.request.urlopen(ref.image_url) as resp:
                tmp.write(resp.read())
            tmp.close()
            path = tmp.name
        else:
            continue

        frame_idx = ref.frame_idx
        if frame_idx < 0:
            frame_idx = max(0, num_frames + frame_idx)

        images.append(
            ImageConditioningInput(path=path, frame_idx=frame_idx, strength=1.0)
        )
    return images


JobTask = tuple[str, str, str, str, list[InputReference], dict[str, Any]]
job_queue: asyncio.Queue[JobTask] = asyncio.Queue()


@torch.inference_mode()
def run_generation_task(
    video_id: str,
    prompt: str,
    seconds_str: str,
    size_str: str,
    input_refs: list[InputReference],
    **kwargs: Any,
) -> None:
    job = JOBS.get(video_id)
    if not job:
        return

    job["status"] = "in_progress"
    job["progress"] = 10

    try:
        try:
            w, h = map(int, size_str.lower().split("x"))
        except Exception:
            w, h = 1280, 764

        frame_rate = kwargs.get("frame_rate", 24.0)
        num_frames = int(float(seconds_str) * frame_rate) + 1
        images = prepare_images(input_refs, num_frames)

        job["progress"] = 30
        hdr = resolve_hdr_color_space(images=images, hdr=kwargs.get("hdr", None))
        vae_dtype = vae_dtype_for_hdr(hdr, torch.bfloat16)

        video, audio, resolved_frames, tiling_config = pipeline(
            prompt=prompt,
            seed=kwargs.get("seed", random.randint(0, 2**31 - 1)),
            height=h,
            width=w,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            vae_dtype=vae_dtype,
            color_space=hdr,
            tiling_config=AUTO_TILING,
            enhance_prompt=kwargs.get("enhance_prompt", False),
            enhance_static_cache=kwargs.get("enhance_static_cache", False),
        )

        job["progress"] = 80
        out_path = OUTPUT_DIR / f"{video_id}.mp4"
        encode_video(
            video=video,
            fps=frame_rate,
            audio=audio,
            output_path=str(out_path),
            video_chunks_number=get_video_chunks_number(resolved_frames, tiling_config),
            color_space=hdr,
        )

        job["status"] = "completed"
        job["progress"] = 100
        job["completed_at"] = int(time.time())
        job["output_path"] = str(out_path)

    except Exception as err:
        logger.exception("Generation error for %s", video_id)
        job["status"] = "failed"
        job["error"] = {"code": "generation_failed", "message": str(err)}


async def queue_worker() -> None:
    while True:
        video_id, prompt, seconds_str, size_str, input_refs, kwargs = (
            await job_queue.get()
        )
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                run_generation_task,
                video_id,
                prompt,
                seconds_str,
                size_str,
                input_refs,
                **kwargs,
            )
        except Exception:
            logger.exception("Error processing generation task for %s", video_id)
        finally:
            job_queue.task_done()


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker_task = asyncio.create_task(queue_worker())
    yield
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="LTX-2 OpenAI Compatible Video Generation API",
    lifespan=lifespan,
)


# Endpoints
@app.post("/videos", response_model=VideoObject)
@app.post("/v1/videos", response_model=VideoObject)
async def create_video(req: CreateVideoRequest) -> VideoObject:
    refs = req.input_references or (
        [req.input_reference] if req.input_reference else []
    )
    video_id = f"video_{uuid.uuid4().hex[:12]}"
    extra_body = req.__pydantic_extra__ or {}

    job = {
        "id": video_id,
        "object": "video",
        "created_at": int(time.time()),
        "completed_at": None,
        "expires_at": None,
        "status": "queued",
        "progress": 0,
        "model": req.model,
        "prompt": req.prompt,
        "seconds": str(req.seconds),
        "size": req.size,
        "remixed_from_video_id": None,
        "error": None,
        "output_path": None,
    }
    JOBS[video_id] = job

    job_queue.put_nowait(
        (video_id, req.prompt, str(req.seconds), req.size, refs, extra_body)
    )

    return VideoObject(**job)


@app.get("/videos/{video_id}", response_model=VideoObject)
@app.get("/v1/videos/{video_id}", response_model=VideoObject)
async def get_video(video_id: str) -> VideoObject:
    if video_id not in JOBS:
        raise HTTPException(status_code=404, detail="Video not found")
    return VideoObject(**JOBS[video_id])


@app.delete("/videos/{video_id}", response_model=DeleteVideoResponse)
@app.delete("/v1/videos/{video_id}", response_model=DeleteVideoResponse)
async def delete_video(video_id: str) -> DeleteVideoResponse:
    if video_id not in JOBS:
        raise HTTPException(status_code=404, detail="Video not found")

    job = JOBS.pop(video_id)
    out_path = job.get("output_path")
    if out_path and os.path.exists(out_path):
        try:
            os.remove(out_path)
        except OSError:
            pass

    return DeleteVideoResponse(id=video_id, deleted=True)


@app.get("/videos", response_model=VideoListResponse)
@app.get("/v1/videos", response_model=VideoListResponse)
async def list_videos(
    after: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    order: str = Query(default="desc"),
) -> VideoListResponse:
    jobs_list = sorted(
        [VideoObject(**j) for j in JOBS.values()],
        key=lambda x: x.created_at,
        reverse=(order.lower() == "desc"),
    )

    if after:
        for idx, item in enumerate(jobs_list):
            if item.id == after:
                jobs_list = jobs_list[idx + 1 :]
                break

    page = jobs_list[:limit]
    return VideoListResponse(
        object="list",
        data=page,
        first_id=page[0].id if page else None,
        last_id=page[-1].id if page else None,
        has_more=len(jobs_list) > limit,
    )


@app.get("/videos/{video_id}/content")
@app.get("/v1/videos/{video_id}/content")
async def get_video_content(
    video_id: str, variant: str = Query(default="video")
) -> FileResponse:
    job = JOBS.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video not found")
    if job.get("status") != "completed" or not job.get("output_path"):
        raise HTTPException(status_code=400, detail="Video generation not completed")

    return FileResponse(
        path=job["output_path"], media_type="video/mp4", filename=f"{video_id}.mp4"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=GLOBAL_ARGS.host, port=GLOBAL_ARGS.port)
