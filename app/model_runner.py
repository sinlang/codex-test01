from __future__ import annotations

from abc import ABC, abstractmethod

from app.config import ModelConfig


class PredictorAdapter(ABC):
    @abstractmethod
    def predict_batch(self, features: list[dict[str, float]]) -> list[float]:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_version(self) -> str:
        raise NotImplementedError


class TorchEasyRecPredictorAdapter(PredictorAdapter):
    """TorchEasyRec Predictor 的封装示例。

    生产环境中可在 __init__ 里加载 torcheasyrec 导出的模型，
    在 predict_batch 中调用其 batch 推理接口。
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._model_version = f"{config.model_name}:v1"

    def predict_batch(self, features: list[dict[str, float]]) -> list[float]:
        # 这里使用可解释的 mock 分数逻辑：
        # score = 0.6*user_ctr_7d + 0.2*user_cvr_7d + 0.2*item_ctr_7d
        outputs: list[float] = []
        for row in features:
            score = (
                0.6 * float(row.get("user_ctr_7d", 0.0))
                + 0.2 * float(row.get("user_cvr_7d", 0.0))
                + 0.2 * float(row.get("item_ctr_7d", 0.0))
            )
            outputs.append(score)
        return outputs

    @property
    def model_version(self) -> str:
        return self._model_version
