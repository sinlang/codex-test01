from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query

from app.config import load_config
from app.feature_store import InMemoryFeatureStore
from app.model_runner import TorchEasyRecPredictorAdapter
from app.realtime_batcher import MicroBatcher
from app.schemas import RecommendRequest
from app.service import RealtimeInferenceService

config = load_config()

feature_store = InMemoryFeatureStore()
predictor = TorchEasyRecPredictorAdapter(config.model)
batcher = MicroBatcher(
    predictor=predictor,
    max_batch_size=config.model.max_batch_size,
    max_wait_ms=config.model.max_wait_ms,
)
service = RealtimeInferenceService(feature_store=feature_store, predictor=predictor, batcher=batcher)

app = FastAPI(title="TorchEasyRec Online Inference", version="0.2.0")


@app.on_event("shutdown")
def shutdown_event() -> None:
    batcher.close()


@app.get("/healthz")
def healthz() -> dict[str, str | int]:
    return {
        "status": "ok",
        "model": predictor.model_version,
        "max_batch_size": config.model.max_batch_size,
    }


@app.post("/v1/recommend")
def recommend(payload: dict, topk: int = Query(default=20, ge=1, le=200)) -> dict:
    try:
        req = RecommendRequest.from_payload(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    topk = min(topk, len(req.candidates))
    try:
        resp = service.recommend(req, topk=topk)
        return asdict(resp)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"inference_error: {exc}") from exc
