"""仅本机默认运行的网页API。静态文件与API同源，不依赖外部CDN。"""
from __future__ import annotations
import csv
import hashlib
import json
import math
import threading
from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from core.kb.graph_io import case_from_dict
from core.kb.rules import build_rules
from core.kb.rules_dsl import guard_vars
from core.kb.guard_display import describe_guard
from .schemas import CaseInput, AnalyzeRequest, QueryRequest, LabOneTask, LabTwoTask, MAX_BODY_BYTES, MAX_ENTITIES, MAX_RELATIONS
from .service import InferenceService, WebError, run_bounded
from .worker import preset_case, model_catalog
from .schemas import GovernanceInput, SaveGovernanceRequest, LoadGovernanceRequest
from core.kb.governance import governance_snapshot
from core.kb.history import GovernanceHistory, RevisionConflict

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "equity-reasoning-web"
WORKSPACE_ID = hashlib.sha256(str(ROOT).casefold().encode("utf-8")).hexdigest()[:16]


def _error(code, message, status=400, details=None):
    return JSONResponse({"error": {"code": code, "message": message, "details": details or []}}, status_code=status)


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as file:
        data = list(csv.DictReader(file))
    for row in data:
        for key, value in list(row.items()):
            if value == "":
                row[key] = None
            elif value in ("True", "False"):
                row[key] = value == "True"
            else:
                try:
                    number = float(value)
                    if math.isfinite(number):
                        row[key] = number
                except (ValueError, TypeError):
                    pass
    return data


def experiment_data():
    e2e = ROOT / "outputs" / "lab02" / "end_to_end"
    metadata = {}
    manifest = e2e / "manifest.json"
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8-sig"))
            metadata = {key: data.get(key) for key in ("created_utc", "config", "decision_note", "zero_violation_note", "label_note")}
        except (ValueError, OSError):
            pass
    return {"available": (e2e / "summary.csv").is_file(), "metadata": metadata,
            "summary": _read_csv(e2e / "summary.csv"), "paired": _read_csv(e2e / "paired_summary.csv"),
            "benchmark": _read_csv(ROOT / "outputs" / "lab02" / "benchmarks" / "forward_summary.csv"),
            "indicator_summary": _read_csv(ROOT / "outputs" / "lab02" / "experiments" / "summary.csv"),
            "sensitivity": _read_csv(ROOT / "outputs" / "lab02" / "experiments" / "sensitivity.csv"),
            "conflict_summary": _read_csv(ROOT / "outputs" / "lab02" / "experiments" / "conflict_summary.csv"),
            "goal_slicing": _read_csv(ROOT / "outputs" / "lab02" / "benchmarks" / "goal_slicing.csv"),
            "resolution": _read_csv(ROOT / "outputs" / "lab02" / "benchmarks" / "resolution.csv"),
            "note": "读取本地保存的实测结果。全部分类标签均为合成标签，不代表真实企业监管准确率。"}


async def _body(request: Request, schema):
    media = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media != "application/json":
        raise WebError("content_type", "请使用JSON格式提交数据", 415)
    if request.headers.get("content-encoding", "identity") != "identity":
        raise WebError("encoding", "不接受压缩的请求体", 415)
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY_BYTES:
            raise WebError("too_large", f"输入超过{MAX_BODY_BYTES // 1024}KB，请精简案例", 413)
        chunks.append(chunk)
    try:
        value = json.loads(b"".join(chunks).decode("utf-8-sig"))
    except (ValueError, UnicodeError) as exc:
        raise WebError("invalid_json", "JSON无法解析，请检查文件格式", 400) from exc
    try:
        return schema.model_validate(value)
    except ValidationError as exc:
        details = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]]
        raise WebError("invalid_input", "输入校验未通过：" + "；".join(details), 422) from exc


def create_app(service: InferenceService | None = None, governance_database: Path | None = None, governance_runner=None) -> FastAPI:
    app = FastAPI(title="股权推理工作台", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.inference = service or InferenceService()
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]", "testserver"])
    app.add_middleware(GZipMiddleware, minimum_size=1200)

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin:
            parsed = urlparse(origin)
            if parsed.scheme not in ("http", "https") or parsed.hostname not in ("localhost", "127.0.0.1", "::1", "testserver"):
                return _error("origin_rejected", "只接受本机网页发起的推理请求", 403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "same-origin"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'; object-src 'none'; base-uri 'self'"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(WebError)
    async def web_error(_request, exc):
        return _error(exc.code, exc.message, exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, exc):
        return _error("invalid_input", "请求参数不正确", 422)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "app": APP_NAME, "workspace_id": WORKSPACE_ID}

    course_slots = threading.BoundedSemaphore(2)
    def course_task(operation, payload):
        if not course_slots.acquire(blocking=False):
            raise WebError("busy", "已有两个实验任务运行，请稍后重试", 429)
        try:
            return run_bounded(operation, payload, timeout=12)
        finally:
            course_slots.release()

    @app.get("/api/course")
    def course_catalog():
        from .course import catalog
        return catalog()

    @app.get("/api/course/files/{lab}/{identifier}")
    def course_file(lab: str, identifier: str):
        from .course import file_path
        try:
            path = file_path(lab, identifier)
        except KeyError as exc:
            raise WebError("not_found", str(exc), 404) from exc
        except FileNotFoundError as exc:
            raise WebError("not_available", str(exc), 404) from exc
        return FileResponse(path, filename=path.name)

    @app.get("/api/labs/lab01/snapshot")
    async def lab_one_snapshot():
        return await run_in_threadpool(course_task, "lab01", {"task":"snapshot","source":"P1","target":"C3"})

    @app.post("/api/labs/lab01/run")
    async def lab_one_run(request: Request):
        model = await _body(request, LabOneTask)
        return await run_in_threadpool(course_task, "lab01", model.model_dump())

    @app.post("/api/labs/lab02/run")
    async def lab_two_run(request: Request):
        model = await _body(request, LabTwoTask)
        payload = model.model_dump()
        if payload["model_source"] not in {r["id"] for r in model_catalog()}:
            raise WebError("model_unavailable", "所选参数来源不可用", 422)
        try:
            case_from_dict(payload["case"])
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise WebError("invalid_case", str(exc), 422) from exc
        return await run_in_threadpool(course_task, "lab02", payload)

    @app.get("/api/bootstrap")
    def bootstrap():
        return {"cases": [{"id": "base", "name": "同源案例", "detail": "4名自然人 · 12家企业"},
                          {"id": "extended", "name": "扩展案例", "detail": "9名自然人 · 20家企业"}],
                "models": model_catalog(), "default_case": "base", "default_model": "manual",
                "limits": {"entities": MAX_ENTITIES, "relations": MAX_RELATIONS, "bytes": MAX_BODY_BYTES},
                "methods": [{"id": "hybrid_locked", "name": "混合推理 · 保留结构"},
                            {"id": "hybrid_unlocked", "name": "混合推理 · 不锁定"},
                            {"id": "hybrid_calibrated", "name": "混合推理 · 校准复核（保留证明）"},
                            {"id": "bayes_only", "name": "Bayes评分器"}, {"id": "fuzzy_only", "name": "Mamdani评分器"},
                            {"id": "rules_only", "name": "结构规则"}]}

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: str):
        try:
            case = preset_case(case_id)
        except ValueError as exc:
            raise WebError("not_found", str(exc), 404) from exc
        return {"id": case_id, "case": case, "completeness": {k: True for k in ("holdings", "guarantees", "trades", "registry")}}

    @app.post("/api/cases/validate")
    async def validate_case(request: Request):
        model = await _body(request, CaseInput)
        payload = model.model_dump()
        try:
            case_from_dict(payload["case"])
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise WebError("invalid_case", str(exc)[:500], 422) from exc
        return {"id": "custom", **payload}

    @app.post("/api/analyze")
    async def analyze(request: Request):
        model = await _body(request, AnalyzeRequest)
        payload = model.model_dump()
        if payload["model_source"] not in {r["id"] for r in model_catalog()}:
            raise WebError("model_unavailable", "该模型预设不可用，请选择教学预设", 422)
        try:
            case_from_dict(payload["case"])
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            raise WebError("invalid_case", str(exc)[:500], 422) from exc
        return await run_in_threadpool(app.state.inference.analyze, payload)

    @app.get("/api/analyses/{identifier}")
    def get_analysis(identifier: str):
        return app.state.inference.get(identifier)["data"]

    @app.post("/api/query")
    async def query(request: Request):
        model = await _body(request, QueryRequest)
        return await run_in_threadpool(app.state.inference.query, model.analysis_id, model.predicate, model.arguments)

    @app.get("/api/rules")
    def rules():
        return {"items": [{"name": r.name, "description": r.desc, "group": r.group, "priority": r.priority,
                           "antecedents": [str(p) for p in r.antecedents], "consequent": str(r.consequent),
                           "guard_variables": sorted(guard_vars(r.guard)), "has_guard": r.guard is not None,
                            **describe_guard(r.guard), "note": r.note}
                          for r in build_rules()]}

    @app.get("/api/experiments")
    def experiments():
        return experiment_data()

    governance_slots = threading.BoundedSemaphore(2)
    def calculate_governance(payload):
        if not governance_slots.acquire(blocking=False):
            raise WebError("busy", "已有两个证据核查在进行，请稍候重试", 429)
        try:
            return governance_runner(payload) if governance_runner else run_bounded("governance", payload, timeout=12)
        finally:
            governance_slots.release()

    def governance_store():
        return GovernanceHistory(governance_database or ROOT / ".state" / "governance.sqlite3", runner=calculate_governance)

    @app.get("/api/governance/example")
    def governance_example():
        from core.kb.governance_examples import cyclic_example
        return cyclic_example()

    @app.get("/api/governance/experiments")
    def governance_experiments():
        return {"rows": _read_csv(ROOT / "outputs" / "lab02" / "governance" / "equity_summary.csv"),
                "note": "目标上游循环权益与独立迭代核对；实际耗时，不是现实法律认证"}

    @app.get("/api/governance/cases")
    def governance_cases():
        return {"items": governance_store().list_cases(),
                "note": "本机持久化版本；凭证核验是人工声明，不是权威认证"}

    @app.post("/api/governance/preview")
    async def governance_preview(request: Request):
        data = await _body(request, GovernanceInput)
        try:
            return await run_in_threadpool(calculate_governance, data.model_dump())
        except (ValueError, KeyError, TypeError) as exc:
            raise WebError("invalid_governance", str(exc)[:600], 422) from exc

    @app.post("/api/governance/import")
    async def governance_import(request: Request):
        data = await _body(request, GovernanceInput)
        try:
            payload = data.model_dump()
            snapshot = await run_in_threadpool(calculate_governance, payload)
            return {"payload": payload, "snapshot": snapshot, "persisted": False}
        except (ValueError, KeyError, TypeError) as exc:
            raise WebError("invalid_governance", str(exc)[:600], 422) from exc

    @app.post("/api/governance/save")
    async def governance_save(request: Request):
        data = await _body(request, SaveGovernanceRequest)
        try:
            return await run_in_threadpool(governance_store().save, data.name, data.payload.model_dump(),
                                           data.case_id, data.expected_revision)
        except RevisionConflict as exc:
            raise WebError("revision_conflict", str(exc), 409) from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise WebError("invalid_governance", str(exc)[:600], 422) from exc

    @app.post("/api/governance/load")
    async def governance_load(request: Request):
        data = await _body(request, LoadGovernanceRequest)
        try:
            return await run_in_threadpool(governance_store().evaluate, data.case_id, data.as_of, data.revision)
        except KeyError as exc:
            raise WebError("not_found", "保存的案例不存在", 404) from exc
        except (ValueError, TypeError) as exc:
            raise WebError("invalid_governance", str(exc)[:600], 422) from exc

    @app.api_route("/api/{missing:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    def api_not_found(missing: str):
        raise WebError("not_found", "接口不存在", 404)

    dist = ROOT / "web" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    else:
        @app.get("/")
        def frontend_missing():
            return _error("frontend_missing", "网页尚未构建。请在web目录执行npm install和npm run build。", 503)
    return app
