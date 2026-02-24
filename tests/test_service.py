from __future__ import annotations

import unittest

from app.config import ModelConfig
from app.feature_store import InMemoryFeatureStore
from app.model_runner import TorchEasyRecPredictorAdapter
from app.schemas import RecommendRequest
from app.service import RealtimeInferenceService


class ServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = RealtimeInferenceService(
            feature_store=InMemoryFeatureStore(),
            predictor=TorchEasyRecPredictorAdapter(ModelConfig()),
        )

    def test_recommend_returns_ranked_items(self) -> None:
        req = RecommendRequest(
            user_id="u_1001",
            scene="homepage",
            candidates=["item_c", "item_b", "item_a"],
            context_features={"hour": 9},
        )
        resp = self.service.recommend(req, topk=2)

        self.assertEqual(resp.user_id, "u_1001")
        self.assertEqual(len(resp.items), 2)
        self.assertGreaterEqual(resp.items[0].score, resp.items[1].score)
        self.assertEqual(resp.items[0].item_id, "item_a")

    def test_recommend_dedup_candidates(self) -> None:
        req = RecommendRequest(
            user_id="u_1001",
            scene="homepage",
            candidates=["item_a", "item_b", "item_a"],
        )
        resp = self.service.recommend(req, topk=10)
        self.assertEqual([item.item_id for item in resp.items], ["item_a", "item_b"])

    def test_request_from_payload_validation(self) -> None:
        with self.assertRaises(ValueError):
            RecommendRequest.from_payload({"user_id": "", "scene": "home", "candidates": ["a"]})
        with self.assertRaises(ValueError):
            RecommendRequest.from_payload({"user_id": "u1", "scene": "", "candidates": ["a"]})
        with self.assertRaises(ValueError):
            RecommendRequest.from_payload({"user_id": "u1", "scene": "home", "candidates": []})


if __name__ == "__main__":
    unittest.main()
