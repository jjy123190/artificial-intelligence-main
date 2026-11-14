"""可回放的股权图JSON接口；只读输入，不解析代码、不读取标签字段。"""
from __future__ import annotations
from dataclasses import asdict
from .domain import Entity, Holding, Guarantee, Trade, Kinship, EquityCase

SCHEMA_VERSION = 1


def case_to_dict(case: EquityCase) -> dict:
    case.validate()
    return {"schema_version": SCHEMA_VERSION, **asdict(case)}


def case_from_dict(data: dict) -> EquityCase:
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("股权图需要 schema_version=1 的对象")
    permitted = {"schema_version", "entities", "holdings", "guarantees", "trades", "kinships", "source", "label"}
    if set(data) - permitted:
        raise ValueError(f"股权图出现未知字段: {sorted(set(data) - permitted)}")
    constructors = {"entities": Entity, "holdings": Holding, "guarantees": Guarantee,
                    "trades": Trade, "kinships": Kinship}
    parsed = {}
    for name, constructor in constructors.items():
        raw = data.get(name, [])
        if not isinstance(raw, list):
            raise ValueError(f"{name} 必须为数组")
        try:
            parsed[name] = [constructor(**item) for item in raw]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 条目格式错误") from exc
    if not isinstance(data.get("source", ""), str) or not isinstance(data.get("label", ""), str):
        raise ValueError("source/label 必须为字符串")
    case = EquityCase(**parsed, source=data.get("source", ""), label=data.get("label", ""))
    try:
        return case.validate()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"股权图校验失败: {exc}") from exc
