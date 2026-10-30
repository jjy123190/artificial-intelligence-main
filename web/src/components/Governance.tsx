import ExperimentBreadcrumb from "./ExperimentBreadcrumb";
import { useEffect, useRef, useState } from "react";
import {
  ArrowClockwise,
  DownloadSimple,
  UploadSimple,
  FloppyDisk,
  Warning,
  FolderSimple,
} from "@phosphor-icons/react";
import type { CaseData } from "../types";
import { request, downloadJson, canonical } from "../api";
import ActionMenu from "./ActionMenu";
import ConfirmDiscard from "./ConfirmDiscard";
import type { Confirmation } from "./ConfirmDiscard";

type Document = {
  id: string;
  title: string;
  content: string;
  sha256: string;
  status: "pending" | "verified" | "revoked";
  reviewer: string;
  valid_from: string;
  valid_to: string;
};
type Identity = {
  entity_id: string;
  identifier: string;
  status: "pending" | "verified" | "rejected";
  reviewer: string;
  evidence_id: string;
};
type Relation = {
  key: string;
  evidence_id: string;
  valid_from: string;
  valid_to: string;
};
type Right = {
  id: string;
  party: string;
  target: string;
  kind: "votes" | "agreement";
  share: number;
  evidence_id: string;
  valid_from: string;
  valid_to: string;
};
type Payload = {
  case: CaseData;
  as_of: string;
  target: string;
  strict_evidence: boolean;
  strict_identity: boolean;
  documents: Document[];
  identities: Identity[];
  relations: Relation[];
  rights: Right[];
};
type Fact = (string | number)[];
type Snapshot = {
  as_of: string;
  effective_case: CaseData;
  claims: Fact[];
  proofs: { fact: Fact; lines: string[] }[];
  documents: (Document & { usable: boolean })[];
  identities: (Identity & { usable: boolean })[];
  excluded_relations: { key: string; reason: string }[];
  retained_relations: Relation[];
  conflicts: { target: string; reason: string; right_ids: string[] }[];
  blocked_rights: { id: string; reason: string }[];
  equity: {
    status: string;
    reason?: string;
    owners: { entity_id: string; name: string; share: number }[];
    unknown_share: number | null;
    residual: number | null;
    companies_solved: number;
    note: string;
  };
  stats: { facts: number; elapsed_ms: number };
  note: string;
};
type Changes = {
  added: Fact[];
  withdrawn: Fact[];
  unchanged: number;
  strategy: string;
};
type Revision = {
  revision: number;
  name: string;
  created_utc: string;
  hash: string;
};
type Saved = {
  case_id: string;
  revision: number;
  name: string;
  payload: Payload;
  snapshot: Snapshot;
  changes: Changes;
  history: Revision[];
  persisted: boolean;
};
type EquityBenchmark = {
  companies: string;
  elapsed_ms_mean: string;
  elapsed_ms_sd: string;
  reference_error_mean: string;
  residual_mean: string;
};
type Listing = {
  case_id: string;
  revision: number;
  name: string;
  created_utc: string;
};
const today = () => {
  const d = new Date();
  return [
    d.getFullYear(),
    String(d.getMonth() + 1).padStart(2, "0"),
    String(d.getDate()).padStart(2, "0"),
  ].join("-");
};
const initial = (value: CaseData): Payload => ({
  case: structuredClone(value),
  as_of: today(),
  target: value.entities.some((e) => e.eid === "C3")
    ? "C3"
    : value.entities.find((e) => e.kind === "Company")!.eid,
  strict_evidence: false,
  strict_identity: false,
  documents: [],
  identities: [],
  relations: [],
  rights: [],
});
const factText = (fact: Fact) =>
  String(fact[0]) + "(" + fact.slice(1).join(", ") + ")";
function delta(before: Snapshot | null, after: Snapshot): Changes {
  const a = new Set(before?.claims.map(canonical) || []);
  const b = new Set(after.claims.map(canonical));
  return {
    added: after.claims.filter((f) => !a.has(canonical(f))),
    withdrawn: (before?.claims || []).filter((f) => !b.has(canonical(f))),
    unchanged: [...a].filter((f) => b.has(f)).length,
    strategy: "rebuild_from_active_evidence",
  };
}

export default function Governance({
  currentCase,
  active = true,
}: {
  currentCase: CaseData | null;
  active?: boolean;
}) {
  const [payload, setPayload] = useState<Payload | null>(
    currentCase ? initial(currentCase) : null,
  );
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [storedSignature, setStoredSignature] = useState("");
  const [confirmation, setConfirmation] = useState<Confirmation | null>(null);
  const confirmationRef = useRef<((accepted: boolean) => void) | null>(null);
  const [view, setView] = useState<
    "overview" | "evidence" | "relations" | "history"
  >("overview");
  const [changes, setChanges] = useState<Changes | null>(null);
  const [accepted, setAccepted] = useState("");
  const [listing, setListing] = useState<Listing[]>([]);
  const [savedId, setSavedId] = useState("");
  const [revision, setRevision] = useState(0);
  const [history, setHistory] = useState<Revision[]>([]);
  const [name, setName] = useState("股权证据核查案例");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [followToday, setFollowToday] = useState(true);
  const [benchmarks, setBenchmarks] = useState<EquityBenchmark[]>([]);
  const [benchmarkError, setBenchmarkError] = useState("");
  const [pendingReplacement, setPendingReplacement] = useState<{
    payload: Payload;
    name: string;
    snapshot: Snapshot | null;
    notice: string;
  } | null>(null);
  const [relationKey, setRelationKey] = useState("");
  const [relationEvidence, setRelationEvidence] = useState("");
  const [relationFrom, setRelationFrom] = useState("");
  const [relationTo, setRelationTo] = useState("");
  const [identityId, setIdentityId] = useState("P1");
  const [identityCode, setIdentityCode] = useState("");
  const [identityReviewer, setIdentityReviewer] = useState("");
  const [identityEvidence, setIdentityEvidence] = useState("");
  const [rightParty, setRightParty] = useState("P1");
  const [rightKind, setRightKind] = useState<Right["kind"]>("votes");
  const [rightShare, setRightShare] = useState("60");
  const [rightEvidence, setRightEvidence] = useState("");
  const [rightFrom, setRightFrom] = useState("");
  const [rightTo, setRightTo] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const caseFileRef = useRef<HTMLInputElement>(null);
  const requestRef = useRef<AbortController | null>(null);
  const operationBusy = useRef(false);
  const liveRef = useRef({
    busy,
    payload,
    savedId,
    revision,
    followToday,
    accepted,
    snapshot,
    active,
  });
  liveRef.current = {
    busy,
    payload,
    savedId,
    revision,
    followToday,
    accepted,
    snapshot,
    active,
  };
  const dirty = !!payload && accepted !== canonical(payload);
  const storageSignature = payload ? canonical({ name, payload }) : "";
  const unsaved = !!storedSignature && storageSignature !== storedSignature;
  useEffect(() => {
    if (!storedSignature && storageSignature)
      setStoredSignature(storageSignature);
  }, [storedSignature, storageSignature]);
  useEffect(() => {
    if (!unsaved) return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [unsaved]);
  function resolveConfirmation(accepted: boolean) {
    const resolve = confirmationRef.current;
    confirmationRef.current = null;
    setConfirmation(null);
    resolve?.(accepted);
  }
  function exportDraft() {
    if (!payload) return;
    downloadJson({ schema_version: 1, name, payload }, "股权核查案例.json");
    setStoredSignature(storageSignature);
    setNotice("已发起草稿下载，请确认文件已保存。核查与保存版本是不同操作。");
  }
  const refreshListing = async () => {
    const result = await request<{ items: Listing[] }>("/api/governance/cases");
    setListing(result.items);
  };
  useEffect(() => {
    if (!payload && currentCase) setPayload(initial(currentCase));
  }, [currentCase, payload]);
  useEffect(() => {
    void refreshListing().catch((e) => setError(e.message));
    const controller = new AbortController();
    void request<{ rows: EquityBenchmark[] }>("/api/governance/experiments", {
      signal: controller.signal,
    })
      .then((result) => setBenchmarks(result.rows))
      .catch((e) => {
        if (!controller.signal.aborted)
          setBenchmarkError(
            e instanceof Error ? e.message : "实测结果读取失败",
          );
      });
    return () => {
      controller.abort();
      requestRef.current?.abort();
      confirmationRef.current?.(false);
      confirmationRef.current = null;
    };
  }, []);
  useEffect(() => {
    const timer = setInterval(() => {
      const state = liveRef.current;
      if (
        !state.active ||
        !state.followToday ||
        state.busy ||
        operationBusy.current ||
        !!confirmationRef.current ||
        !state.payload ||
        state.accepted !== canonical(state.payload)
      )
        return;
      if (state.payload.as_of !== today()) {
        setPayload((p) => p && { ...p, as_of: today() });
        setNotice(
          "日期已变化，请重新计算当前有效证据；旧结果不会冒充最新结论。",
        );
        const next = { ...state.payload, as_of: today() };
        requestRef.current?.abort();
        const controller = new AbortController();
        requestRef.current = controller;
        operationBusy.current = true;
        setBusy(true);
        void request<Snapshot>("/api/governance/preview", {
          method: "POST",
          signal: controller.signal,
          body: JSON.stringify(next),
        })
          .then((result) => {
            if (controller.signal.aborted) return;
            setChanges(delta(state.snapshot, result));
            setSnapshot(result);
            setAccepted(canonical(next));
            setNotice(
              "日期已变化，已自动重新核查当前草稿的有效证据并撤回失效结论。保存的旧版本不会被覆盖。",
            );
          })
          .catch((e) => {
            if (!controller.signal.aborted)
              setError(e instanceof Error ? e.message : "自动更新失败");
          })
          .finally(() => {
            if (requestRef.current === controller) {
              operationBusy.current = false;
              setBusy(false);
            }
          });
      }
    }, 30000);
    return () => clearInterval(timer);
  }, []);
  const patch = (update: Partial<Payload>) => {
    if (operationBusy.current) return;
    setPayload((p) => p && { ...p, ...update });
  };
  const updateDocument = (index: number, value: Partial<Document>) => {
    if (!payload) return;
    patch({
      documents: payload.documents.map((d, i) =>
        i === index
          ? {
              ...d,
              ...value,
              ...(value.content !== undefined
                ? { sha256: "", status: "pending" as const }
                : {}),
            }
          : d,
      ),
    });
  };
  async function calculate(save: boolean) {
    if (!payload || operationBusy.current || confirmationRef.current) return;
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    operationBusy.current = true;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      if (save) {
        const result = await request<Saved>("/api/governance/save", {
          method: "POST",
          signal: controller.signal,
          body: JSON.stringify({
            name,
            payload,
            case_id: savedId || null,
            expected_revision: revision,
          }),
        });
        if (controller.signal.aborted) return;
        setSnapshot(result.snapshot);
        setChanges(result.changes);
        setSavedId(result.case_id);
        setRevision(result.revision);
        setStoredSignature(canonical({ name, payload }));
        setHistory(result.history);
        setView("overview");
        setAccepted(canonical(payload));
        try {
          await refreshListing();
        } catch {
          setError(
            "新版本已保存，但案例目录刷新失败。请稍后重新打开历史与实验；不要重复保存同一版本。",
          );
        }
        if (controller.signal.aborted) return;
        setNotice(
          "版本 " +
            result.revision +
            " 已保存到本机，历史证据和证明不会被覆盖。",
        );
      } else {
        const result = await request<Snapshot>("/api/governance/preview", {
          method: "POST",
          signal: controller.signal,
          body: JSON.stringify(payload),
        });
        if (controller.signal.aborted) return;
        setChanges(delta(snapshot, result));
        setSnapshot(result);
        setAccepted(canonical(payload));
        setView("overview");
        setNotice("当前有效证据已重新推理；失效结论不再出现在新快照中。");
      }
    } catch (e) {
      if (!controller.signal.aborted)
        setError(e instanceof Error ? e.message : "操作失败");
    } finally {
      if (requestRef.current === controller) {
        operationBusy.current = false;
        setBusy(false);
      }
    }
  }
  async function load(
    id: string,
    selectedRevision?: number,
    date = payload?.as_of || today(),
  ) {
    if (operationBusy.current || confirmationRef.current) return;
    if (unsaved) {
      const accepted = await new Promise<boolean>((resolve) => {
        confirmationRef.current = resolve;
        setConfirmation({
          title: "加载历史版本？",
          description:
            "当前核查草稿有尚未保存或导出的修改。加载会替换这些内容，已保存的历史不会删除。",
          action: "加载历史版本",
        });
      });
      if (!accepted || operationBusy.current) return;
    }
    requestRef.current?.abort();
    const controller = new AbortController();
    requestRef.current = controller;
    operationBusy.current = true;
    setBusy(true);
    setError("");
    try {
      const result = await request<Saved>("/api/governance/load", {
        method: "POST",
        signal: controller.signal,
        body: JSON.stringify({
          case_id: id,
          revision: selectedRevision ?? null,
          as_of: date,
        }),
      });
      if (controller.signal.aborted) return;
      setPayload(result.payload);
      setAccepted(canonical(result.payload));
      setSnapshot(result.snapshot);
      setChanges(result.changes);
      setSavedId(result.case_id);
      setRevision(result.history.at(-1)?.revision || result.revision);
      setHistory(result.history);
      setName(result.name);
      setStoredSignature(
        canonical({ name: result.name, payload: result.payload }),
      );
      setView("overview");
      setNotice(
        "已加载版本 " +
          result.revision +
          "，按 " +
          date +
          " 自动核对证据有效期。后续保存追加到最新版本。",
      );
    } catch (e) {
      if (!controller.signal.aborted)
        setError(e instanceof Error ? e.message : "加载失败");
    } finally {
      if (requestRef.current === controller) {
        operationBusy.current = false;
        setBusy(false);
      }
    }
  }
  async function addDocument(file?: File) {
    if (!payload || operationBusy.current || confirmationRef.current) return;
    operationBusy.current = true;
    setBusy(true);
    try {
      if (file && (file.size > 16000 || !/\.(txt|md)$/i.test(file.name)))
        throw new Error(
          "仅支持16KB以内的TXT/Markdown文本凭证；不自动认定扫描件真实性。",
        );
      const content = file ? await file.text() : "";
      const sha256 = file
        ? Array.from(
            new Uint8Array(
              await crypto.subtle.digest(
                "SHA-256",
                new TextEncoder().encode(content),
              ),
            ),
          )
            .map((b) => b.toString(16).padStart(2, "0"))
            .join("")
        : "";
      setPayload(
        (current) =>
          current && {
            ...current,
            documents: [
              ...current.documents,
              {
                id: "ev-" + crypto.randomUUID().slice(0, 8),
                title: file?.name || "核验凭证",
                content,
                sha256,
                status: "pending",
                reviewer: "",
                valid_from: "",
                valid_to: "",
              },
            ],
          },
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : "凭证读取失败");
    } finally {
      operationBusy.current = false;
      setBusy(false);
    }
  }
  function replaceDraft(next: {
    payload: Payload;
    name: string;
    snapshot: Snapshot | null;
    notice: string;
  }) {
    requestRef.current?.abort();
    setPayload(next.payload);
    setName(next.name);
    setStoredSignature(canonical({ name: next.name, payload: next.payload }));
    setSnapshot(next.snapshot);
    setChanges(next.snapshot ? delta(null, next.snapshot) : null);
    setAccepted(next.snapshot ? canonical(next.payload) : "");
    setSavedId("");
    setRevision(0);
    setHistory([]);
    setFollowToday(next.payload.as_of === today());
    setRelationKey("");
    setRelationEvidence("");
    setRelationFrom("");
    setRelationTo("");
    const first = next.payload.case.entities[0]?.eid || "";
    setIdentityId(first);
    setIdentityCode("");
    setIdentityReviewer("");
    setIdentityEvidence("");
    setRightParty(first);
    setRightEvidence("");
    setRightFrom("");
    setRightTo("");
    setRightKind("votes");
    setRightShare("60");
    setPendingReplacement(null);
    setView("overview");
    setError("");
    setNotice(next.notice);
    operationBusy.current = false;
    setBusy(false);
  }
  function proposeReplacement(next: {
    payload: Payload;
    name: string;
    snapshot: Snapshot | null;
    notice: string;
  }) {
    if (
      payload &&
      (snapshot ||
        savedId ||
        payload.documents.length ||
        payload.relations.length ||
        payload.identities.length ||
        payload.rights.length)
    ) {
      setPendingReplacement(next);
      setNotice("");
    } else replaceDraft(next);
  }
  async function importCase(file: File) {
    if (operationBusy.current || confirmationRef.current) return;
    const controller = new AbortController();
    requestRef.current?.abort();
    requestRef.current = controller;
    operationBusy.current = true;
    setBusy(true);
    setError("");
    setNotice("");
    try {
      if (file.size > 262144 || !/\.json$/i.test(file.name))
        throw new Error("请选择256KB以内的JSON核查案例。");
      const value: unknown = JSON.parse(await file.text());
      if (!value || typeof value !== "object" || Array.isArray(value))
        throw new Error("文件必须包含核查案例对象。");
      const envelope = value as {
        payload?: unknown;
        name?: unknown;
        schema_version?: unknown;
      };
      if (
        envelope.schema_version !== undefined &&
        envelope.schema_version !== 1
      )
        throw new Error("不支持此核查案例格式版本。");
      const candidate = envelope.payload ?? value;
      const result = await request<{ payload: Payload; snapshot: Snapshot }>(
        "/api/governance/import",
        {
          method: "POST",
          signal: controller.signal,
          body: JSON.stringify(candidate),
        },
      );
      if (controller.signal.aborted) return;
      proposeReplacement({
        ...result,
        name:
          typeof envelope.name === "string" && envelope.name.trim()
            ? envelope.name.trim().slice(0, 120)
            : "导入核查案例",
        notice:
          "已导入并重新核查有效证据。文件附带的旧结论和历史指纹不被信任；保存后创建新的本机案例。",
      });
    } catch (e) {
      if (!controller.signal.aborted)
        setError(e instanceof Error ? e.message : "核查案例导入失败");
    } finally {
      if (requestRef.current === controller) {
        operationBusy.current = false;
        setBusy(false);
      }
    }
  }
  async function loadCycleExample() {
    if (operationBusy.current || confirmationRef.current) return;
    const controller = new AbortController();
    requestRef.current?.abort();
    requestRef.current = controller;
    operationBusy.current = true;
    setBusy(true);
    setError("");
    try {
      const result = await request<{
        case: CaseData;
        target: string;
        note: string;
      }>("/api/governance/example", { signal: controller.signal });
      if (controller.signal.aborted) return;
      proposeReplacement({
        payload: { ...initial(result.case), target: result.target },
        name: "循环持股权益教学案例",
        snapshot: null,
        notice: result.note + "。请点击重新核查，查看实际求解结果。",
      });
    } catch (e) {
      if (!controller.signal.aborted)
        setError(e instanceof Error ? e.message : "教学案例加载失败");
    } finally {
      if (requestRef.current === controller) {
        operationBusy.current = false;
        setBusy(false);
      }
    }
  }
  if (!active) return null;
  if (!payload)
    return (
      <main id="main-content" tabIndex={-1} className="page-shell">
        <ExperimentBreadcrumb moduleId="modeling" title="证据与更新（扩展）" />
        <h1>证据与更新</h1>
        <p>正在等待案例载入……</p>
      </main>
    );
  const entities = payload.case.entities;
  const relationOptions = (
    ["holdings", "guarantees", "trades", "kinships"] as const
  ).flatMap((kind) =>
    payload.case[kind].map((r) => {
      const values = r as unknown as Record<string, string | number>;
      const from =
        values.holder || values.guarantor || values.seller || values.a;
      const to =
        values.target || values.beneficiary || values.buyer || values.b;
      return {
        key: kind + ":" + from + ":" + to,
        label:
          {
            holdings: "持股",
            guarantees: "担保",
            trades: "交易",
            kinships: "亲属",
          }[kind] +
          " · " +
          from +
          " → " +
          to,
      };
    }),
  );
  const evidenceOptions = (
    <>
      <option value="">未指定凭证</option>
      {payload.documents.map((d) => (
        <option value={d.id} key={d.id}>
          {d.title} · {d.id}
        </option>
      ))}
    </>
  );
  return (
    <main
      id={active ? "main-content" : undefined}
      tabIndex={-1}
      className="page-shell governance-page"
    >
      <ConfirmDiscard
        confirmation={confirmation}
        onResolve={resolveConfirmation}
        onExport={exportDraft}
      />
      <ExperimentBreadcrumb moduleId="modeling" title="证据与更新（扩展）" />
      <header className="page-header">
        <div>
          <h1>证据与更新</h1>
          <p>核验记录、日期快照与版本追溯。经济权益和表决权分开计算。</p>
        </div>
        <button
          className="button secondary"
          disabled={!snapshot || busy || dirty}
          onClick={() => {
            downloadJson(
              { schema_version: 1, name, payload, snapshot, changes, history },
              "股权证据与版本.json",
            );
            setStoredSignature(storageSignature);
          }}
        >
          <DownloadSimple size={17} />
          导出核查结果
        </button>
      </header>
      <p className="governance-disclosure">
        <Warning size={18} />
        核验状态由操作者人工登记，不是外部权威认证。这里的案例版本会保存到本机；请勿上传真实身份证等敏感信息。
      </p>
      {unsaved && (
        <p className="governance-storage-state" role="status">
          核查草稿尚未保存或导出。重新核查只更新预览；保存新版本才会写入本机。
        </p>
      )}
      {error && (
        <div className="error-banner" role="alert">
          {error}
        </div>
      )}
      {notice && (
        <p className="governance-notice" role="status">
          {notice}
        </p>
      )}
      <section className="governance-controls" aria-label="日期与版本控制">
        <label>
          案例名称
          <input
            value={name}
            maxLength={120}
            onChange={(e) => setName(e.target.value)}
            disabled={busy}
          />
        </label>
        <label>
          核查日期
          <input
            aria-label="核查日期"
            type="date"
            value={payload.as_of}
            onChange={(e) => {
              patch({ as_of: e.target.value });
              setFollowToday(false);
            }}
            disabled={busy}
          />
        </label>
        <label>
          目标企业
          <select
            aria-label="核查目标企业"
            value={payload.target}
            onChange={(e) => patch({ target: e.target.value })}
            disabled={busy}
          >
            {entities
              .filter((e) => e.kind === "Company")
              .map((e) => (
                <option key={e.eid} value={e.eid}>
                  {e.name} · {e.eid}
                </option>
              ))}
          </select>
        </label>
        <ActionMenu
          label="核查选项"
          id="governance-options"
          className="governance-options"
        >
          <p className="menu-heading">有效证据使用方式</p>
          <label className="governance-check">
            <input
              type="checkbox"
              checked={followToday}
              disabled={busy}
              onChange={(e) => {
                setFollowToday(e.target.checked);
                if (e.target.checked) patch({ as_of: today() });
              }}
            />
            跟随今天（页面打开时每30秒核对日期）
          </label>
          <label className="governance-check">
            <input
              type="checkbox"
              checked={payload.strict_evidence}
              disabled={busy}
              onChange={(e) => patch({ strict_evidence: e.target.checked })}
            />
            只使用凭证已核验的关系
          </label>
          <label className="governance-check">
            <input
              type="checkbox"
              checked={payload.strict_identity}
              disabled={busy}
              onChange={(e) => patch({ strict_identity: e.target.checked })}
            />
            关系主体必须完成人工身份核验
          </label>
        </ActionMenu>
        <div className="page-actions">
          <button
            className="button primary"
            disabled={busy}
            onClick={() => void calculate(false)}
          >
            <ArrowClockwise size={17} />
            {busy ? "正在核查" : "重新核查有效证据"}
          </button>
          <button
            className="button secondary"
            disabled={busy || !name.trim()}
            onClick={() => void calculate(true)}
          >
            <FloppyDisk size={17} />
            保存新版本
          </button>
        </div>
        {dirty && snapshot && (
          <p className="stale-notice">
            证据或日期已修改，下面是旧快照。请重新核查，旧结论不作为当前结果。
          </p>
        )}
      </section>
      <section
        className="governance-case-tools"
        aria-label="核查案例导入与演示"
      >
        <p>
          当前案例：{payload.case.label || name} · {entities.length}{" "}
          个实体。核查草稿独立于工作台，替换草稿不会改动原始数据或已保存版本。
        </p>
        <ActionMenu
          label="核查案例"
          icon={<FolderSimple size={17} />}
          id="governance-case-menu"
          className="governance-case-menu"
        >
          <p className="menu-heading">切换、导入与导出</p>
          <button
            className="button secondary"
            disabled={busy || !currentCase}
            onClick={() =>
              currentCase &&
              proposeReplacement({
                payload: initial(currentCase),
                name: "工作台证据核查案例",
                snapshot: null,
                notice:
                  "已重新接入当前工作台案例。原核验草稿没有自动迁移，请重新核查或登记凭证。",
              })
            }
          >
            接入当前工作台案例
          </button>
          <button
            className="button secondary"
            disabled={busy}
            onClick={() => void loadCycleExample()}
          >
            加载循环持股教学案例
          </button>
          <button
            className="button secondary"
            disabled={busy}
            onClick={() => caseFileRef.current?.click()}
          >
            <UploadSimple size={17} />
            导入核查案例JSON
          </button>
          <button
            className="button secondary"
            disabled={busy}
            onClick={exportDraft}
          >
            <DownloadSimple size={17} />
            导出核查案例
          </button>
          <input
            ref={caseFileRef}
            className="sr-only"
            type="file"
            accept=".json,application/json"
            aria-label="导入核查案例JSON"
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void importCase(f);
              e.target.value = "";
            }}
          />
        </ActionMenu>
        {pendingReplacement && (
          <div className="governance-replacement" role="alert">
            <p>
              即将用“{pendingReplacement.name}
              ”替换当前草稿。未保存内容和当前预览将被替换，已保存的本机历史版本不会删除。
            </p>
            <div className="page-actions">
              <button
                className="button primary"
                disabled={busy}
                onClick={() => replaceDraft(pendingReplacement)}
              >
                确认替换草稿
              </button>
              <button
                className="button secondary"
                disabled={busy}
                onClick={() => setPendingReplacement(null)}
              >
                保留当前草稿
              </button>
            </div>
          </div>
        )}
      </section>
      <div className="governance-tabs" role="tablist" aria-label="证据工作区">
        {(["overview", "evidence", "relations", "history"] as const).map(
          (tab, i) => (
            <button
              key={tab}
              role="tab"
              id={"governance-tab-" + tab}
              aria-selected={view === tab}
              aria-controls={"governance-panel-" + tab}
              tabIndex={view === tab ? 0 : -1}
              onClick={() => setView(tab)}
              onKeyDown={(event) => {
                if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
                  event.preventDefault();
                  const next = (
                    ["overview", "evidence", "relations", "history"] as const
                  )[(i + (event.key === "ArrowRight" ? 1 : 3)) % 4];
                  setView(next);
                  document.getElementById("governance-tab-" + next)?.focus();
                }
              }}
            >
              {
                {
                  overview: "核查结果",
                  evidence: "凭证与身份",
                  relations: "关系与协议",
                  history: "历史与实验",
                }[tab]
              }
            </button>
          ),
        )}
      </div>
      <div
        role="tabpanel"
        id="governance-panel-overview"
        aria-labelledby="governance-tab-overview"
        hidden={view !== "overview"}
      >
        {!snapshot && (
          <section className="governance-welcome">
            <span className="welcome-icon">
              <FolderSimple size={28} />
            </span>
            <h2>从当前证据，建立可追溯的结论</h2>
            <p>
              直接核查当前案例，或先登记凭证、设置关系有效期。结论、撤回变化与经济权益会在这里显示。
            </p>
            <div>
              <button
                className="button primary"
                disabled={busy}
                onClick={() => void calculate(false)}
              >
                开始核查
              </button>
              <button
                className="button secondary"
                onClick={() => setView("evidence")}
              >
                登记凭证
              </button>
            </div>
          </section>
        )}
        {snapshot && (
          <section
            className={
              "governance-section governance-results" + (dirty ? " stale" : "")
            }
            aria-label="有效证据核查结果"
          >
            <h2>当前快照与撤回结果</h2>
            <p>
              快照日期 {snapshot.as_of} · {snapshot.stats.facts} 条事实 · 排除{" "}
              {snapshot.excluded_relations.length} 条失效或未核验关系
            </p>
            {snapshot.conflicts.map((c) => (
              <p className="governance-conflict" role="status" key={c.target}>
                {c.target}：{c.reason}。相关权利不用于控制推断。
              </p>
            ))}
            {snapshot.excluded_relations.length > 0 && (
              <details>
                <summary>查看失效关系</summary>
                <ul>
                  {snapshot.excluded_relations.map((r) => (
                    <li key={r.key}>
                      {r.key}：{r.reason}
                    </li>
                  ))}
                </ul>
              </details>
            )}
            {snapshot.blocked_rights.map((r) => (
              <p key={r.id}>
                {r.id}：{r.reason}
              </p>
            ))}
            <div className="governance-change-grid">
              <div>
                <h3>新增结论：{changes?.added.length || 0}</h3>
                <ul>
                  {changes?.added.map((f) => (
                    <li key={canonical(f)}>{factText(f)}</li>
                  ))}
                </ul>
              </div>
              <div>
                <h3>撤回结论：{changes?.withdrawn.length || 0}</h3>
                <ul aria-label="撤回结论列表">
                  {changes?.withdrawn.map((f) => (
                    <li key={canonical(f)}>{factText(f)}</li>
                  ))}
                </ul>
              </div>
            </div>
            <details>
              <summary>有效快照全部证明（{snapshot.proofs.length}）</summary>
              {snapshot.proofs.map((p) => (
                <div className="governance-proof" key={canonical(p.fact)}>
                  <strong>{factText(p.fact)}</strong>
                  <pre>{p.lines.join("\n")}</pre>
                </div>
              ))}
            </details>
            <h3>循环持股经济权益</h3>
            <p>{snapshot.equity.note}</p>
            {snapshot.equity.status !== "solved" ? (
              <p className="governance-conflict">
                无法唯一求解：{snapshot.equity.reason}
              </p>
            ) : (
              <div className="table-scroll">
                <table aria-label="经济权益计算结果">
                  <thead>
                    <tr>
                      <th>自然人</th>
                      <th>经济权益比例</th>
                    </tr>
                  </thead>
                  <tbody>
                    {snapshot.equity.owners.map((o) => (
                      <tr key={o.entity_id}>
                        <td>
                          {o.name} · {o.entity_id}
                        </td>
                        <td>{(o.share * 100).toFixed(4)}%</td>
                      </tr>
                    ))}
                    <tr>
                      <td>未知股东</td>
                      <td>
                        {((snapshot.equity.unknown_share || 0) * 100).toFixed(
                          4,
                        )}
                        %
                      </td>
                    </tr>
                  </tbody>
                </table>
                <p>
                  求解 {snapshot.equity.companies_solved} 家上游企业 · 方程残差{" "}
                  {snapshot.equity.residual?.toExponential(2)}
                </p>
              </div>
            )}
          </section>
        )}
      </div>
      <div
        role="tabpanel"
        id="governance-panel-evidence"
        aria-labelledby="governance-tab-evidence"
        hidden={view !== "evidence"}
      >
        <section className="governance-section">
          <h2>来源凭证</h2>
          <p>
            先登记内容和核验人，再确认状态。修改内容会回到“待核验”；SHA256用于检查内容是否改变，不证明内容真实。
          </p>
          <div className="page-actions">
            <button
              className="button secondary"
              disabled={busy}
              onClick={() => void addDocument()}
            >
              添加文本凭证
            </button>
            <button
              className="button secondary"
              disabled={busy}
              onClick={() => fileRef.current?.click()}
            >
              <UploadSimple size={17} />
              导入TXT凭证
            </button>
            <input
              ref={fileRef}
              className="sr-only"
              type="file"
              accept=".txt,.md"
              aria-label="导入文本核验凭证"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) void addDocument(f);
                e.target.value = "";
              }}
            />
          </div>
          {!payload.documents.length && (
            <p className="governance-empty">
              还没有凭证。可以先预览原教学案例，或登记一份文本凭证并绑定关系。
            </p>
          )}
          {payload.documents.map((d, i) => (
            <fieldset
              className="governance-document"
              key={d.id}
              disabled={busy}
            >
              <legend>
                凭证 {i + 1} · {d.id}
              </legend>
              <div className="governance-form-grid">
                <label>
                  凭证标题
                  <input
                    value={d.title}
                    maxLength={160}
                    onChange={(e) =>
                      updateDocument(i, { title: e.target.value })
                    }
                  />
                </label>
                <label>
                  核验人
                  <input
                    aria-label={"凭证核验人 " + (i + 1)}
                    value={d.reviewer}
                    maxLength={80}
                    onChange={(e) =>
                      updateDocument(i, { reviewer: e.target.value })
                    }
                  />
                </label>
                <label>
                  凭证状态
                  <select
                    aria-label={"凭证状态 " + (i + 1)}
                    value={d.status}
                    onChange={(e) =>
                      updateDocument(i, {
                        status: e.target.value as Document["status"],
                      })
                    }
                  >
                    <option value="pending">待核验</option>
                    <option value="verified">人工已核验</option>
                    <option value="revoked">已撤销</option>
                  </select>
                </label>
                <label>
                  生效日
                  <input
                    type="date"
                    value={d.valid_from}
                    onChange={(e) =>
                      updateDocument(i, { valid_from: e.target.value })
                    }
                  />
                </label>
                <label>
                  失效日（当天不再有效）
                  <input
                    aria-label={"凭证失效日 " + (i + 1)}
                    type="date"
                    value={d.valid_to}
                    onChange={(e) =>
                      updateDocument(i, { valid_to: e.target.value })
                    }
                  />
                </label>
              </div>
              <label>
                凭证文本
                <textarea
                  aria-label={"凭证文本 " + (i + 1)}
                  value={d.content}
                  maxLength={16000}
                  rows={3}
                  onChange={(e) =>
                    updateDocument(i, { content: e.target.value })
                  }
                />
              </label>
              <p className="governance-hash">
                SHA256：
                {(!dirty &&
                  snapshot?.documents.find((x) => x.id === d.id)?.sha256) ||
                  d.sha256 ||
                  "重新核查时生成"}
              </p>
            </fieldset>
          ))}
        </section>
        <section className="governance-section">
          <h2>人工身份登记</h2>
          <p>
            统一标识用来发现同一身份被绑定到多个实体的冲突，不自动合并或认证身份。
          </p>
          <div className="governance-form-grid">
            <label>
              核验主体
              <select
                aria-label="核验主体"
                value={identityId}
                onChange={(e) => setIdentityId(e.target.value)}
              >
                {entities.map((e) => (
                  <option key={e.eid} value={e.eid}>
                    {e.name} · {e.eid}
                  </option>
                ))}
              </select>
            </label>
            <label>
              核验标识
              <input
                value={identityCode}
                maxLength={120}
                onChange={(e) => setIdentityCode(e.target.value)}
                placeholder="使用虚构编号，勿填真实证件号"
              />
            </label>
            <label>
              身份核验人
              <input
                value={identityReviewer}
                maxLength={80}
                onChange={(e) => setIdentityReviewer(e.target.value)}
              />
            </label>
            <label>
              身份凭证
              <select
                aria-label="身份凭证"
                value={identityEvidence}
                onChange={(e) => setIdentityEvidence(e.target.value)}
              >
                {evidenceOptions}
              </select>
            </label>
            <button
              className="button secondary"
              disabled={
                busy ||
                !identityEvidence ||
                !identityCode.trim() ||
                !identityReviewer.trim()
              }
              onClick={() =>
                patch({
                  identities: [
                    ...payload.identities.filter(
                      (r) => r.entity_id !== identityId,
                    ),
                    {
                      entity_id: identityId,
                      identifier: identityCode,
                      reviewer: identityReviewer,
                      evidence_id: identityEvidence,
                      status: "verified",
                    },
                  ],
                })
              }
            >
              登记人工身份核验
            </button>
          </div>
          <ul className="governance-records">
            {payload.identities.map((r) => (
              <li key={r.entity_id}>
                {r.entity_id} · {r.identifier} · 核验人 {r.reviewer} ·{" "}
                {r.evidence_id}
                <button
                  className="button secondary"
                  disabled={busy}
                  onClick={() =>
                    patch({
                      identities: payload.identities.map((x) =>
                        x.entity_id === r.entity_id
                          ? { ...x, status: "rejected" }
                          : x,
                      ),
                    })
                  }
                >
                  撤销身份确认
                </button>
              </li>
            ))}
          </ul>
        </section>
      </div>
      <div
        role="tabpanel"
        id="governance-panel-relations"
        aria-labelledby="governance-tab-relations"
        hidden={view !== "relations"}
      >
        <section className="governance-section">
          <h2>关系有效期与凭证</h2>
          <div className="governance-form-grid">
            <label>
              选择关系
              <select
                aria-label="选择有效期关系"
                value={relationKey}
                onChange={(e) => {
                  setRelationKey(e.target.value);
                  const r = payload.relations.find(
                    (x) => x.key === e.target.value,
                  );
                  setRelationEvidence(r?.evidence_id || "");
                  setRelationFrom(r?.valid_from || "");
                  setRelationTo(r?.valid_to || "");
                }}
              >
                <option value="">请选择</option>
                {relationOptions.map((r) => (
                  <option key={r.key} value={r.key}>
                    {r.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              关系凭证
              <select
                aria-label="关系凭证"
                value={relationEvidence}
                onChange={(e) => setRelationEvidence(e.target.value)}
              >
                {evidenceOptions}
              </select>
            </label>
            <label>
              关系生效日
              <input
                type="date"
                value={relationFrom}
                onChange={(e) => setRelationFrom(e.target.value)}
              />
            </label>
            <label>
              关系失效日
              <input
                aria-label="关系失效日"
                type="date"
                value={relationTo}
                onChange={(e) => setRelationTo(e.target.value)}
              />
            </label>
            <button
              className="button secondary"
              disabled={busy || !relationKey}
              onClick={() =>
                patch({
                  relations: [
                    ...payload.relations.filter((r) => r.key !== relationKey),
                    {
                      key: relationKey,
                      evidence_id: relationEvidence,
                      valid_from: relationFrom,
                      valid_to: relationTo,
                    },
                  ],
                })
              }
            >
              更新关系设置
            </button>
          </div>
          <ul className="governance-records">
            {payload.relations.map((r) => (
              <li key={r.key}>
                {r.key} · {r.valid_from || "不限开始"} →{" "}
                {r.valid_to || "不限结束"} · {r.evidence_id || "无凭证"}
              </li>
            ))}
          </ul>
        </section>
        <section className="governance-section">
          <h2>表决权与控制协议</h2>
          <p>
            明确权利记录优先于“股权比例等于表决权”的教学假设。冲突记录将被阻断并要求人工裁决；过期后重新依据有效证据推理。
          </p>
          <div className="governance-form-grid">
            <label>
              权利主体
              <select
                aria-label="权利主体"
                value={rightParty}
                onChange={(e) => setRightParty(e.target.value)}
              >
                {entities
                  .filter((e) => e.eid !== payload.target)
                  .map((e) => (
                    <option key={e.eid} value={e.eid}>
                      {e.name} · {e.eid}
                    </option>
                  ))}
              </select>
            </label>
            <label>
              权利类型
              <select
                aria-label="权利类型"
                value={rightKind}
                onChange={(e) => setRightKind(e.target.value as Right["kind"])}
              >
                <option value="votes">明确表决权</option>
                <option value="agreement">控制协议（互斥）</option>
              </select>
            </label>
            {rightKind === "votes" && (
              <label>
                表决权百分比
                <input
                  type="number"
                  min={0}
                  max={100}
                  step={1}
                  value={rightShare}
                  onChange={(e) => setRightShare(e.target.value)}
                />
              </label>
            )}
            <label>
              权利凭证
              <select
                aria-label="权利凭证"
                value={rightEvidence}
                onChange={(e) => setRightEvidence(e.target.value)}
              >
                {evidenceOptions}
              </select>
            </label>
            <label>
              权利生效日
              <input
                type="date"
                value={rightFrom}
                onChange={(e) => setRightFrom(e.target.value)}
              />
            </label>
            <label>
              权利失效日
              <input
                type="date"
                value={rightTo}
                onChange={(e) => setRightTo(e.target.value)}
              />
            </label>
            <button
              className="button secondary"
              disabled={
                busy ||
                !rightEvidence ||
                rightParty === payload.target ||
                (rightKind === "votes" &&
                  (rightShare === "" ||
                    !Number.isFinite(Number(rightShare)) ||
                    Number(rightShare) < 0 ||
                    Number(rightShare) > 100))
              }
              onClick={() =>
                patch({
                  rights: [
                    ...payload.rights,
                    {
                      id: "right-" + crypto.randomUUID().slice(0, 8),
                      party: rightParty,
                      target: payload.target,
                      kind: rightKind,
                      share:
                        rightKind === "votes" ? Number(rightShare) / 100 : 0,
                      evidence_id: rightEvidence,
                      valid_from: rightFrom,
                      valid_to: rightTo,
                    },
                  ],
                })
              }
            >
              添加权利记录
            </button>
          </div>
          <ul className="governance-records">
            {payload.rights.map((r) => (
              <li key={r.id}>
                {r.party} → {r.target} ·{" "}
                {r.kind === "votes"
                  ? (r.share * 100).toFixed(1) + "%表决权"
                  : "控制协议"}{" "}
                · {r.evidence_id}
                <button
                  className="button secondary"
                  disabled={busy}
                  onClick={() =>
                    patch({
                      rights: payload.rights.filter((x) => x.id !== r.id),
                    })
                  }
                >
                  删除草稿权利
                </button>
              </li>
            ))}
          </ul>
        </section>
      </div>
      <div
        role="tabpanel"
        id="governance-panel-history"
        aria-labelledby="governance-tab-history"
        hidden={view !== "history"}
      >
        <section className="governance-section">
          <h2>本机案例与历史版本</h2>
          <p>
            凭证和历史存放在本项目 .state
            目录，不会覆盖原始CSV。关闭网页后仍保留；不要把它当QA缓存删除。
          </p>
          <label>
            打开保存的案例
            <select
              aria-label="打开保存的核查案例"
              value={savedId}
              disabled={busy}
              onChange={(e) => {
                if (e.target.value) void load(e.target.value);
              }}
            >
              <option value="">请选择已保存案例</option>
              {listing.map((r) => (
                <option key={r.case_id} value={r.case_id}>
                  {r.name} · 最新版本 {r.revision}
                </option>
              ))}
            </select>
          </label>
          {history.length > 0 && (
            <div className="table-scroll">
              <table aria-label="案例版本历史">
                <thead>
                  <tr>
                    <th>版本</th>
                    <th>保存时间</th>
                    <th>内容指纹</th>
                    <th>操作</th>
                  </tr>
                </thead>
                <tbody>
                  {history.map((r) => (
                    <tr key={r.revision}>
                      <td>{r.revision}</td>
                      <td>{new Date(r.created_utc).toLocaleString()}</td>
                      <td>
                        <code>{r.hash.slice(0, 16)}…</code>
                      </td>
                      <td>
                        <button
                          className="button secondary"
                          disabled={busy}
                          onClick={() => void load(savedId, r.revision)}
                        >
                          加载版本 {r.revision}
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
        <section className="governance-section" aria-label="循环权益扩容实测">
          <h2>循环权益扩容实测</h2>
          <p>
            对8、32、64、128家上游企业的合成循环持股图重复求解，并与独立不动点迭代核对。以下数据读取正式实验文件，不是估算或现实企业认证。
          </p>
          {benchmarkError ? (
            <p role="status">{benchmarkError}</p>
          ) : benchmarks.length === 0 ? (
            <p className="governance-empty">
              暂未找到正式扩容结果。已有案例计算仍可使用；完成正式实验后这里会显示实测值。
            </p>
          ) : (
            <div className="table-scroll">
              <table aria-label="循环权益扩容实测结果">
                <thead>
                  <tr>
                    <th>上游企业数</th>
                    <th>求解平均ms</th>
                    <th>标准差ms</th>
                    <th>独立迭代误差</th>
                    <th>方程残差</th>
                  </tr>
                </thead>
                <tbody>
                  {benchmarks.map((r) => (
                    <tr key={r.companies}>
                      <td>{r.companies}</td>
                      <td>{Number(r.elapsed_ms_mean).toFixed(4)}</td>
                      <td>{Number(r.elapsed_ms_sd).toFixed(4)}</td>
                      <td>{Number(r.reference_error_mean).toExponential(2)}</td>
                      <td>{Number(r.residual_mean).toExponential(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p>
            100%封闭循环返回“无法唯一求解”，不会强造权益数值。经济权益与表决权、控制协议的法律效力是不同问题。
          </p>
        </section>
      </div>
    </main>
  );
}
