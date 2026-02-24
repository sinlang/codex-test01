from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


@dataclass
class ModelConfig:
    model_dir: str = "./model_export"
    model_name: str = "torcheasyrec_ranker"
    max_batch_size: int = 32
    max_wait_ms: int = 3


@dataclass
class ServiceConfig:
    host: str = "0.0.0.0"
    port: int = 8080
    request_timeout_ms: int = 60


@dataclass
class FeatureConfig:
    online_features: list[str] = field(
        default_factory=lambda: ["user_ctr_7d", "user_cvr_7d", "item_ctr_7d"]
    )


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)


def load_config() -> AppConfig:
    cfg = AppConfig()
    cfg.model.model_dir = os.getenv("MODEL_DIR", cfg.model.model_dir)
    cfg.model.model_name = os.getenv("MODEL_NAME", cfg.model.model_name)
    cfg.model.max_batch_size = int(os.getenv("MAX_BATCH_SIZE", str(cfg.model.max_batch_size)))
    cfg.model.max_wait_ms = int(os.getenv("MAX_WAIT_MS", str(cfg.model.max_wait_ms)))
    cfg.service.port = int(os.getenv("PORT", str(cfg.service.port)))
    return cfg


def validate_model_dir(model_dir: str) -> bool:
    return Path(model_dir).exists()
