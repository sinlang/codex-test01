from __future__ import annotations

from abc import ABC, abstractmethod


class FeatureStoreAdapter(ABC):
    @abstractmethod
    def get_user_features(self, user_id: str) -> dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def get_item_features(self, item_id: str) -> dict[str, float]:
        raise NotImplementedError


class InMemoryFeatureStore(FeatureStoreAdapter):
    """示例特征存储，生产可替换为 Redis/HBase/FeatureService。"""

    def __init__(self) -> None:
        self.user_table = {
            "u_1001": {"user_ctr_7d": 0.12, "user_cvr_7d": 0.04},
            "u_1002": {"user_ctr_7d": 0.08, "user_cvr_7d": 0.03},
        }
        self.item_table = {
            "item_a": {"item_ctr_7d": 0.20},
            "item_b": {"item_ctr_7d": 0.14},
            "item_c": {"item_ctr_7d": 0.09},
        }

    def get_user_features(self, user_id: str) -> dict[str, float]:
        return self.user_table.get(user_id, {"user_ctr_7d": 0.01, "user_cvr_7d": 0.005})

    def get_item_features(self, item_id: str) -> dict[str, float]:
        return self.item_table.get(item_id, {"item_ctr_7d": 0.01})
