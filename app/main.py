import asyncio
import json
import pathlib
import time
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import (HTMLResponse, JSONResponse, RedirectResponse,
                               Response)

from app.config import ENABLE_MODEL_WARMUP, MAX_BATCH_SIZE, load_model_metadata
from app.logging_config import get_logger
from app.model import model
from app.monitoring import (metrics_payload, record_batch_success,
                            record_invalid_image, record_prediction_error,
                            record_success, set_model_loaded)
from app.postprocessing import postprocess
from app.preprocessing import preprocess_image
from app.visualization import draw_detections

logger = get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model.load()
    if ENABLE_MODEL_WARMUP:
        model.warmup()
    set_model_loaded(model.is_loaded())
    yield


app = FastAPI(
    title="PPE Object Detection API",
    description="Monolithic FastAPI model-as-a-service for YOLO ONNX PPE detection.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return RedirectResponse(url="/demo")


@app.get("/health")
def health() -> dict[str, Any]:
    metadata = load_model_metadata()
    set_model_loaded(model.is_loaded())
    return {
        "status": "ok",
        "model_loaded": model.is_loaded(),
        "model_name": metadata.get("model_name"),
        "model_version": metadata.get("model_version"),
        "model_path": metadata.get("model_path"),
    }


@app.get("/metrics")
def metrics() -> Response:
    set_model_loaded(model.is_loaded())
    return Response(content=metrics_payload(), media_type="text/plain; version=0.0.4")


def validate_confidence_threshold(confidence_threshold: float) -> None:
    if not 0.0 <= confidence_threshold <= 1.0:
        raise HTTPException(
            status_code=422, detail="confidence_threshold must be between 0 and 1"
        )


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.25),
) -> JSONResponse:
    validate_confidence_threshold(confidence_threshold)

    started_at = time.perf_counter()
    try:
        image_bytes = await file.read()
        preprocess_started_at = time.perf_counter()
        input_tensor, original_size, scale, pad, brightness = preprocess_image(
            image_bytes
        )
        preprocess_ms = round((time.perf_counter() - preprocess_started_at) * 1000, 2)

        inference_started_at = time.perf_counter()
        predictions = model.predict(input_tensor)
        inference_ms = round((time.perf_counter() - inference_started_at) * 1000, 2)

        postprocess_started_at = time.perf_counter()
        detections = postprocess(
            predictions, original_size, scale, pad, confidence_threshold
        )
        postprocess_ms = round((time.perf_counter() - postprocess_started_at) * 1000, 2)
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        timing = {
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": latency_ms,
        }

        record_success(detections, original_size, brightness, timing)

        logger.info(
            json.dumps(
                {
                    "event": "prediction",
                    "filename": file.filename,
                    "status": "success",
                    "threshold": confidence_threshold,
                    "image_width": original_size[0],
                    "image_height": original_size[1],
                    "brightness": round(brightness, 2),
                    "timing": timing,
                    "detections": detections,
                }
            )
        )

        return JSONResponse(
            content={
                "status": "success",
                "latency_ms": latency_ms,
                "timing": timing,
                "detections": detections,
            }
        )
    except ValueError as exc:
        record_invalid_image(str(exc))
        logger.warning(
            json.dumps(
                {
                    "event": "prediction",
                    "filename": file.filename,
                    "status": "error",
                    "error_type": "invalid_image",
                    "message": str(exc),
                }
            )
        )
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(exc)},
        )
    except Exception as exc:
        record_prediction_error()
        logger.exception(
            json.dumps(
                {
                    "event": "prediction",
                    "filename": file.filename,
                    "status": "error",
                    "error_type": "prediction_error",
                    "message": str(exc),
                }
            )
        )
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Prediction failed"},
        )


@app.post("/predict-annotated")
async def predict_annotated(
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.25),
) -> Response:
    validate_confidence_threshold(confidence_threshold)

    started_at = time.perf_counter()
    try:
        image_bytes = await file.read()
        preprocess_started_at = time.perf_counter()
        input_tensor, original_size, scale, pad, brightness = preprocess_image(
            image_bytes
        )
        preprocess_ms = round((time.perf_counter() - preprocess_started_at) * 1000, 2)

        inference_started_at = time.perf_counter()
        predictions = model.predict(input_tensor)
        inference_ms = round((time.perf_counter() - inference_started_at) * 1000, 2)

        postprocess_started_at = time.perf_counter()
        detections = postprocess(
            predictions, original_size, scale, pad, confidence_threshold
        )
        annotated_image = draw_detections(image_bytes, detections)
        postprocess_ms = round((time.perf_counter() - postprocess_started_at) * 1000, 2)
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        timing = {
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": latency_ms,
        }

        record_success(detections, original_size, brightness, timing)

        logger.info(
            json.dumps(
                {
                    "event": "annotated_prediction",
                    "filename": file.filename,
                    "status": "success",
                    "threshold": confidence_threshold,
                    "image_width": original_size[0],
                    "image_height": original_size[1],
                    "brightness": round(brightness, 2),
                    "timing": timing,
                    "detection_count": len(detections),
                }
            )
        )

        return Response(
            content=annotated_image,
            media_type="image/jpeg",
            headers={
                "X-Detection-Count": str(len(detections)),
                "X-Latency-Ms": str(latency_ms),
            },
        )
    except ValueError as exc:
        record_invalid_image(str(exc))
        logger.warning(
            json.dumps(
                {
                    "event": "annotated_prediction",
                    "filename": file.filename,
                    "status": "error",
                    "error_type": "invalid_image",
                    "message": str(exc),
                }
            )
        )
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(exc)},
        )
    except Exception as exc:
        record_prediction_error()
        logger.exception(
            json.dumps(
                {
                    "event": "annotated_prediction",
                    "filename": file.filename,
                    "status": "error",
                    "error_type": "prediction_error",
                    "message": str(exc),
                }
            )
        )
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Prediction failed"},
        )


@app.post("/predict-batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    confidence_threshold: float = Form(0.25),
) -> JSONResponse:
    validate_confidence_threshold(confidence_threshold)

    if not files:
        raise HTTPException(
            status_code=422, detail="At least one image file is required"
        )
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Batch size exceeds MAX_BATCH_SIZE={MAX_BATCH_SIZE}",
        )

    started_at = time.perf_counter()
    try:
        preprocess_started_at = time.perf_counter()
        image_stats: list[dict[str, Any]] = []
        input_tensors: list[np.ndarray] = []

        files_data = []
        for file in files:
            image_bytes = await file.read()
            files_data.append((file.filename, image_bytes))

        tasks = [
            asyncio.to_thread(preprocess_image, image_bytes)
            for _, image_bytes in files_data
        ]
        preprocess_results = await asyncio.gather(*tasks)

        for (filename, _), result in zip(files_data, preprocess_results):
            input_tensor, original_size, scale, pad, brightness = result
            input_tensors.append(input_tensor)
            image_stats.append(
                {
                    "filename": filename,
                    "original_size": original_size,
                    "scale": scale,
                    "pad": pad,
                    "brightness": brightness,
                }
            )

        preprocess_ms = round((time.perf_counter() - preprocess_started_at) * 1000, 2)

        inference_started_at = time.perf_counter()
        batch_tensor = np.concatenate(input_tensors, axis=0)
        used_fallback = False
        try:
            batch_predictions = model.predict_batch(batch_tensor)
            if len(batch_predictions) != len(files):
                raise RuntimeError(
                    f"Batch output count {len(batch_predictions)} does not match input count {len(files)}"
                )
        except Exception:
            used_fallback = True
            batch_predictions = [
                model.predict(input_tensor) for input_tensor in input_tensors
            ]

        inference_ms = round((time.perf_counter() - inference_started_at) * 1000, 2)

        postprocess_started_at = time.perf_counter()
        results: list[dict[str, Any]] = []
        for predictions, stats in zip(batch_predictions, image_stats):
            detections = postprocess(
                predictions,
                stats["original_size"],
                stats["scale"],
                stats["pad"],
                confidence_threshold,
            )
            results.append(
                {
                    "filename": stats["filename"],
                    "detections": detections,
                }
            )

        postprocess_ms = round((time.perf_counter() - postprocess_started_at) * 1000, 2)
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        timing = {
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": latency_ms,
        }

        record_batch_success(results, image_stats, timing, used_fallback)

        logger.info(
            json.dumps(
                {
                    "event": "batch_prediction",
                    "status": "success",
                    "batch_size": len(files),
                    "threshold": confidence_threshold,
                    "used_fallback": used_fallback,
                    "timing": timing,
                    "results": results,
                }
            )
        )

        return JSONResponse(
            content={
                "status": "success",
                "batch_size": len(files),
                "latency_ms": latency_ms,
                "timing": timing,
                "used_fallback": used_fallback,
                "results": results,
            }
        )
    except ValueError as exc:
        record_invalid_image(str(exc))
        logger.warning(
            json.dumps(
                {
                    "event": "batch_prediction",
                    "status": "error",
                    "error_type": "invalid_image",
                    "message": str(exc),
                }
            )
        )
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": str(exc)},
        )
    except Exception as exc:
        record_prediction_error()
        logger.exception(
            json.dumps(
                {
                    "event": "batch_prediction",
                    "status": "error",
                    "error_type": "prediction_error",
                    "message": str(exc),
                }
            )
        )
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Prediction failed"},
        )


@app.get("/demo", response_class=HTMLResponse)
def demo_ui() -> HTMLResponse:
    demo_path = pathlib.Path(__file__).parent / "static" / "demo.html"
    if not demo_path.exists():
        raise HTTPException(status_code=404, detail="Demo UI not found")
    return HTMLResponse(content=demo_path.read_text())
