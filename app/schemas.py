from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RecommendRequest:
    user_id: str
    scene: str
    candidates: list[str]
    context_features: dict[str, str | int | float] = field(default_factory=dict)
    request_id: str | None = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RecommendRequest":
        user_id = str(payload.get("user_id", "")).strip()
        scene = str(payload.get("scene", "")).strip()
        raw_candidates = payload.get("candidates", [])
        if not isinstance(raw_candidates, list):
            raise ValueError("candidates must be a list")
        candidates = [str(item).strip() for item in raw_candidates if str(item).strip()]
        if not user_id:
            raise ValueError("user_id must not be empty")
        if not scene:
            raise ValueError("scene must not be empty")
        if not candidates:
            raise ValueError("candidates must not be empty")

        context = payload.get("context_features", {})
        if not isinstance(context, dict):
            raise ValueError("context_features must be a dict")

        request_id = payload.get("request_id")
        normalized_request_id = str(request_id).strip() if request_id is not None else None

        return cls(
            user_id=user_id,
            scene=scene,
            candidates=candidates,
            context_features=context,
            request_id=normalized_request_id or None,
        )


@dataclass
class ScoredItem:
    item_id: str
    score: float


@dataclass
class RecommendResponse:
    user_id: str
    scene: str
    request_id: str
    items: list[ScoredItem]
    model_version: str
