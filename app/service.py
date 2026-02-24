from __future__ import annotations

from dataclasses import dataclass
import uuid

from app.feature_store import FeatureStoreAdapter
from app.model_runner import PredictorAdapter
from app.realtime_batcher import MicroBatcher
from app.schemas import RecommendRequest, RecommendResponse, ScoredItem


@dataclass
class RealtimeInferenceService:
    feature_store: FeatureStoreAdapter
    predictor: PredictorAdapter
    batcher: MicroBatcher | None = None

    def _build_feature(self, user_id: str, item_id: str, context_features: dict[str, str | int | float]) -> dict[str, float]:
        user_feat = self.feature_store.get_user_features(user_id)
        item_feat = self.feature_store.get_item_features(item_id)
        merged: dict[str, float] = {**user_feat, **item_feat}
        for key, value in context_features.items():
            if isinstance(value, (int, float)):
                merged[key] = float(value)
        return merged

    def recommend(self, req: RecommendRequest, topk: int = 20) -> RecommendResponse:
        # 保留候选去重后的原顺序，避免重复 item 拉低吞吐
        seen: set[str] = set()
        dedup_candidates = [item for item in req.candidates if not (item in seen or seen.add(item))]

        if self.batcher is None:
            batch_features = [
                self._build_feature(req.user_id, item_id, req.context_features)
                for item_id in dedup_candidates
            ]
            scores = self.predictor.predict_batch(batch_features)
        else:
            scores = [
                self.batcher.predict(self._build_feature(req.user_id, item_id, req.context_features))
                for item_id in dedup_candidates
            ]

        ranked = sorted(
            [ScoredItem(item_id=item_id, score=score) for item_id, score in zip(dedup_candidates, scores)],
            key=lambda x: x.score,
            reverse=True,
        )[:topk]

        return RecommendResponse(
            user_id=req.user_id,
            scene=req.scene,
            request_id=req.request_id or str(uuid.uuid4()),
            items=ranked,
            model_version=self.predictor.model_version,
        )
