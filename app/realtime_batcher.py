from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import queue
import threading
import time

from app.model_runner import PredictorAdapter


@dataclass
class PredictTask:
    features: dict[str, float]
    future: Future[float]


class MicroBatcher:
    """轻量 micro-batching 执行器。

    - 收到单条样本后入队
    - 后台线程按 max_batch_size 或 max_wait_ms 聚合
    - 聚合后一次性调用 predictor.predict_batch
    """

    def __init__(self, predictor: PredictorAdapter, max_batch_size: int, max_wait_ms: int = 3) -> None:
        self._predictor = predictor
        self._max_batch_size = max(1, max_batch_size)
        self._max_wait_s = max_wait_ms / 1000.0
        self._queue: queue.Queue[PredictTask] = queue.Queue()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._worker.start()

    def predict(self, features: dict[str, float], timeout_s: float = 0.2) -> float:
        fut: Future[float] = Future()
        self._queue.put(PredictTask(features=features, future=fut))
        return fut.result(timeout=timeout_s)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                first = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue

            tasks = [first]
            deadline = time.time() + self._max_wait_s
            while len(tasks) < self._max_batch_size and time.time() < deadline:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    tasks.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break

            self._executor.submit(self._execute_batch, tasks)

    def _execute_batch(self, tasks: list[PredictTask]) -> None:
        try:
            feats = [task.features for task in tasks]
            scores = self._predictor.predict_batch(feats)
            if len(scores) != len(tasks):
                raise RuntimeError("predictor output size mismatch")
            for task, score in zip(tasks, scores):
                task.future.set_result(score)
        except Exception as exc:  # noqa: BLE001
            for task in tasks:
                if not task.future.done():
                    task.future.set_exception(exc)

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=0.5)
        self._executor.shutdown(wait=False, cancel_futures=True)
