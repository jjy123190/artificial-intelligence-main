"""有时限的隔离执行和有界内存缓存；不把上传内容写入原案例文件。"""
from __future__ import annotations
import hashlib
import json
import multiprocessing
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from .worker import worker_entry, MODEL_FILES


class WebError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def run_bounded(operation: str, payload: dict, timeout: float = 12.0) -> dict:
    ctx = multiprocessing.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    process = ctx.Process(target=worker_entry, args=(operation, payload, send), daemon=True)
    try:
        process.start()
        send.close()
        if not receive.poll(timeout):
            raise WebError("timeout", "推理超过网页时间预算，请减少关系或拆分案例后重试", 408)
        try:
            result = json.loads(receive.recv_bytes(maxlength=6 * 1024 * 1024).decode("utf-8"))
        except (EOFError, OSError, ValueError) as exc:
            raise WebError("worker_failed", "推理进程未返回有效结果，请检查案例后重试", 422) from exc
        if not result.get("ok"):
            code = result.get("code", "inference_failed")
            raise WebError(code, result.get("message", "推理未完成"), 400 if code == "invalid_case" else 422)
        return result["data"]
    finally:
        receive.close()
        send.close()
        if process.pid is not None:
            process.join(timeout=.3)
            if process.is_alive():
                # 只结束当前请求自己创建的工作进程，不查找/终止其他Python进程。
                process.terminate()
                process.join(timeout=2)
            process.close()


class InferenceService:
    def __init__(self, *, runner=run_bounded, cache_size: int = 12, ttl: float = 1800):
        self.runner = runner
        self.cache_size, self.ttl = cache_size, ttl
        self._cache = OrderedDict()
        self._keys = {}
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(2)

    def _prune(self):
        now = time.monotonic()
        expired = [key for key, record in self._cache.items() if now - record["time"] > self.ttl]
        for key in expired:
            old = self._cache.pop(key)
            if self._keys.get(old["key"]) == key:
                self._keys.pop(old["key"], None)
        while len(self._cache) > self.cache_size:
            key, old = self._cache.popitem(last=False)
            if self._keys.get(old["key"]) == key:
                self._keys.pop(old["key"], None)

    def analyze(self, payload: dict) -> dict:
        model_state = []
        for key, path in MODEL_FILES.items():
            if payload["model_source"].startswith(key + ":") and path.is_file():
                stat = path.stat()
                model_state = [stat.st_mtime_ns, stat.st_size]
        canonical = json.dumps([payload, model_state], ensure_ascii=False, sort_keys=True, allow_nan=False)
        fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        with self._lock:
            self._prune()
            old_id = self._keys.get(fingerprint)
            if old_id in self._cache:
                self._cache.move_to_end(old_id)
                return {**self._cache[old_id]["data"], "cached": True}
        if not self._slots.acquire(blocking=False):
            raise WebError("busy", "已有推理正在运行，请稍后再试", 429)
        try:
            started = time.perf_counter()
            result = self.runner("analyze", payload)
            identifier = uuid.uuid4().hex
            data = {**result, "id": identifier, "created_at": datetime.now(timezone.utc).isoformat(),
                    "wall_ms": (time.perf_counter() - started) * 1000, "cached": False}
            with self._lock:
                self._cache[identifier] = {"time": time.monotonic(), "key": fingerprint, "data": data, "request": payload}
                self._keys[fingerprint] = identifier
                self._prune()
            return data
        finally:
            self._slots.release()

    def get(self, identifier: str) -> dict:
        with self._lock:
            self._prune()
            if identifier not in self._cache:
                raise WebError("analysis_expired", "这次分析已过期，请重新运行推理", 404)
            self._cache.move_to_end(identifier)
            return self._cache[identifier]

    def query(self, identifier: str, predicate: str, arguments: list[str]) -> dict:
        record = self.get(identifier)
        if not self._slots.acquire(blocking=False):
            raise WebError("busy", "已有推理正在运行，请稍后再试", 429)
        try:
            request = {"case": record["request"]["case"], "completeness": record["request"]["completeness"],
                       "predicate": predicate, "arguments": arguments}
            return self.runner("query", request)
        finally:
            self._slots.release()
