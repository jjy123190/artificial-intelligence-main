"""网页边界的严格输入验证。核心命令行格式保持兼容，网页增加体积/复杂度上限。"""
from __future__ import annotations
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, model_validator

MAX_ENTITIES = 64
MAX_RELATIONS = 180
MAX_BODY_BYTES = 256 * 1024
Identifier = Annotated[StrictStr, Field(min_length=1, max_length=80, pattern=r"^[^?\s][^\r\n]*$")]
Name = Annotated[StrictStr, Field(min_length=1, max_length=100)]
Number = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Unit = Annotated[Number, Field(ge=0, le=1)]
Money = Annotated[Number, Field(ge=0, le=1e12)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EntitySpec(StrictModel):
    eid: Identifier
    name: Name
    kind: Literal["Person", "Company"]
    category: Annotated[StrictStr, Field(max_length=80)] = ""
    capital: Money = 0.0
    opacity: Unit = 0.0
    disclosure: Unit = 1.0


class HoldingSpec(StrictModel):
    holder: Identifier
    target: Identifier
    ratio: Annotated[Number, Field(gt=0, le=1)]
    since: Annotated[StrictStr, Field(max_length=80)] = ""
    layered: StrictBool = False


class GuaranteeSpec(StrictModel):
    guarantor: Identifier
    beneficiary: Identifier
    amount: Money


class TradeSpec(StrictModel):
    seller: Identifier
    buyer: Identifier
    amount: Money


class KinshipSpec(StrictModel):
    a: Identifier
    b: Identifier
    relation: Annotated[StrictStr, Field(min_length=1, max_length=40)]
    evidence: StrictBool = False


class CaseSpec(StrictModel):
    schema_version: Annotated[int, Field(strict=True, ge=1, le=1)] = 1
    entities: Annotated[list[EntitySpec], Field(min_length=1, max_length=MAX_ENTITIES)]
    holdings: Annotated[list[HoldingSpec], Field(max_length=100)] = Field(default_factory=list)
    guarantees: Annotated[list[GuaranteeSpec], Field(max_length=80)] = Field(default_factory=list)
    trades: Annotated[list[TradeSpec], Field(max_length=80)] = Field(default_factory=list)
    kinships: Annotated[list[KinshipSpec], Field(max_length=40)] = Field(default_factory=list)
    source: Annotated[StrictStr, Field(max_length=1200)] = "网页输入"
    label: Annotated[StrictStr, Field(max_length=140)] = "自定义案例"

    @model_validator(mode="after")
    def complexity(self):
        if sum(len(getattr(self, field)) for field in ("holdings", "guarantees", "trades", "kinships")) > MAX_RELATIONS:
            raise ValueError(f"网页演示最多支持{MAX_RELATIONS}条关系")
        if not any(e.kind == "Company" for e in self.entities):
            raise ValueError("案例至少需要一家企业")
        # 持股简单路径预处理有8层上限；限制大型强连通团，避免指数路径枚举拖垮网页。
        adjacent: dict[str, list[str]] = {}
        for h in self.holdings:
            adjacent.setdefault(h.holder, []).append(h.target)
        if any(len(children) > 10 for children in adjacent.values()):
            raise ValueError("网页演示中单个主体最多展示10条对外持股，请精简案例")
        reach = {}
        for entity in self.entities:
            seen, queue = set(), list(adjacent.get(entity.eid, []))
            while queue:
                current = queue.pop()
                if current not in seen:
                    seen.add(current)
                    queue.extend(adjacent.get(current, []))
            reach[entity.eid] = seen
        for origin, reachable in reach.items():
            component = {origin} | {v for v in reachable if origin in reach.get(v, set())}
            if len(component) > 8:
                raise ValueError("持股环路团超过8个实体，网页演示请使用较小案例；命令行仍可单独设置预算")
        return self


class Completeness(StrictModel):
    holdings: StrictBool = True
    trades: StrictBool = True
    guarantees: StrictBool = True
    registry: StrictBool = True


class CaseInput(StrictModel):
    """Validate/normalize source data without running any experiment inference."""
    case: CaseSpec
    completeness: Completeness = Field(default_factory=Completeness)


class AnalyzeRequest(CaseInput):
    method: Literal["rules_only", "bayes_only", "fuzzy_only", "hybrid_unlocked", "hybrid_locked", "hybrid_calibrated"] = "hybrid_locked"
    threshold: Annotated[Number, Field(gt=0, lt=1)] = .5
    model_source: Annotated[StrictStr, Field(min_length=1, max_length=60)] = "manual"


class QueryRequest(StrictModel):
    analysis_id: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{32}$")]
    predicate: Literal["Controls", "ActualController", "Alert", "RiskLevel", "RelatedTrade", "BeneficialOwner"]
    arguments: Annotated[list[Annotated[StrictStr, Field(min_length=1, max_length=100)]], Field(min_length=2, max_length=2)]


# Explicit governance input is separate from legacy graph schemas, preserving old experiment semantics.
class EvidenceDocument(StrictModel):
    id: Annotated[StrictStr, Field(min_length=1, max_length=80)]
    title: Annotated[StrictStr, Field(min_length=1, max_length=160)]
    content: Annotated[StrictStr, Field(max_length=16000)] = ""
    sha256: Annotated[StrictStr, Field(pattern=r"^$|^[a-f0-9]{64}$")] = ""
    status: Literal["pending", "verified", "revoked"] = "pending"
    reviewer: Annotated[StrictStr, Field(max_length=80)] = ""
    valid_from: Annotated[StrictStr, Field(max_length=10)] = ""
    valid_to: Annotated[StrictStr, Field(max_length=10)] = ""


class IdentityRecord(StrictModel):
    entity_id: Identifier
    identifier: Annotated[StrictStr, Field(max_length=120)] = ""
    status: Literal["pending", "verified", "rejected"] = "pending"
    reviewer: Annotated[StrictStr, Field(max_length=80)] = ""
    evidence_id: Annotated[StrictStr, Field(max_length=80)] = ""


class TemporalRelation(StrictModel):
    key: Annotated[StrictStr, Field(min_length=1, max_length=240)]
    evidence_id: Annotated[StrictStr, Field(max_length=80)] = ""
    valid_from: Annotated[StrictStr, Field(max_length=10)] = ""
    valid_to: Annotated[StrictStr, Field(max_length=10)] = ""


class VerifiedRight(StrictModel):
    id: Annotated[StrictStr, Field(min_length=1, max_length=80)]
    party: Identifier
    target: Identifier
    kind: Literal["votes", "agreement"]
    share: Unit = 0.0
    evidence_id: Annotated[StrictStr, Field(min_length=1, max_length=80)]
    valid_from: Annotated[StrictStr, Field(max_length=10)] = ""
    valid_to: Annotated[StrictStr, Field(max_length=10)] = ""


class GovernanceInput(StrictModel):
    case: CaseSpec
    as_of: Annotated[StrictStr, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]
    target: Identifier
    strict_evidence: StrictBool = False
    strict_identity: StrictBool = False
    documents: Annotated[list[EvidenceDocument], Field(max_length=64)] = Field(default_factory=list)
    identities: Annotated[list[IdentityRecord], Field(max_length=64)] = Field(default_factory=list)
    relations: Annotated[list[TemporalRelation], Field(max_length=180)] = Field(default_factory=list)
    rights: Annotated[list[VerifiedRight], Field(max_length=64)] = Field(default_factory=list)


class SaveGovernanceRequest(StrictModel):
    name: Annotated[StrictStr, Field(min_length=1, max_length=120)]
    payload: GovernanceInput
    case_id: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{32}$")] | None = None
    expected_revision: Annotated[int, Field(strict=True, ge=0)] = 0


class LoadGovernanceRequest(StrictModel):
    case_id: Annotated[StrictStr, Field(pattern=r"^[a-f0-9]{32}$")]
    revision: Annotated[int, Field(strict=True, ge=1)] | None = None
    as_of: Annotated[StrictStr, Field(pattern=r"^\d{4}-\d{2}-\d{2}$")]


class LabOneTask(StrictModel):
    task: Literal["snapshot", "rules", "penetration", "path"] = "rules"
    source: Identifier = "P1"
    target: Identifier = "C3"
    max_depth: Annotated[int, Field(strict=True, ge=1, le=10)] = 10


class LabTwoTask(AnalyzeRequest):
    task: Literal["modeling", "deterministic", "resolution", "uncertain", "hybrid"]
    source: Identifier = "P1"
    target: Identifier = "C3"
