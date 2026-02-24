# TorchEasyRec 实时在线推理服务框架（参考实现）

该项目提供了一套可直接用于 **TorchEasyRec** 推荐模型的实时推理服务框架，重点覆盖在线场景必备能力：

- 低延迟单请求推理
- micro-batching 聚合提升吞吐
- 在线特征补全与缺省兜底
- 统一请求校验与错误处理
- 健康检查与可扩展分层设计

## 架构分层

```text
API Layer (FastAPI)
  └── RealtimeInferenceService
       ├── FeatureStoreAdapter
       ├── TorchEasyRecPredictorAdapter
       ├── MicroBatcher
       └── Ranking/PostProcess
```

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### 推理请求示例

```bash
curl -X POST http://127.0.0.1:8080/v1/recommend \
  -H 'Content-Type: application/json' \
  -d '{
    "request_id": "req_001",
    "user_id": "u_1001",
    "scene": "homepage",
    "candidates": ["item_a", "item_b", "item_c", "item_a"],
    "context_features": {"hour": 9, "device": "ios"}
  }'
```

## 目录说明

- `app/config.py`：服务配置定义（含 `MAX_BATCH_SIZE`、`MAX_WAIT_MS`）
- `app/schemas.py`：请求/响应结构与输入校验
- `app/feature_store.py`：在线特征适配层
- `app/model_runner.py`：TorchEasyRec 模型调用封装
- `app/realtime_batcher.py`：micro-batching 执行器
- `app/service.py`：核心在线推理编排逻辑
- `app/main.py`：Web 服务入口
- `tests/test_service.py`：核心服务单元测试

## 生产化建议

1. 将 `InMemoryFeatureStore` 替换为线上特征源（Redis / Feature Platform）。
2. 在 `TorchEasyRecPredictorAdapter` 中接入真实 `torcheasyrec` 导出的模型与 batch API。
3. 对 `recommend` 接口增加限流、熔断、超时和重试策略。
4. 增加监控指标（QPS、P95、错误率、模型耗时、batch 命中率）。
5. 扩展多模型路由（按场景或实验分流到不同模型版本）。
