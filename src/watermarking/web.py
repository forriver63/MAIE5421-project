"""FastAPI web service exposing embed / detect endpoints.

Run with::

    uvicorn watermarking.web:app --host 0.0.0.0 --port 8000

The model is loaded once at startup and reused across requests.
"""
from __future__ import annotations

import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import Response
from PIL import Image

from .pipeline import detect_image, embed_image
from .wam import model as wam_model

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    _model = wam_model.load_default_model()
    yield


app = FastAPI(title="Watermarking service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/embed")
async def embed(image: UploadFile = File(...), payload: int = Form(..., ge=0, le=(1 << 32) - 1)):
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    watermarked = embed_image(img, payload, wam=_model)
    buffer = io.BytesIO()
    watermarked.save(buffer, format="PNG")
    return Response(content=buffer.getvalue(), media_type="image/png")


@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    return detect_image(img, wam=_model)
