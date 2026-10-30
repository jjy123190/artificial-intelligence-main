import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Graph,
  TreeStructure,
  Play,
  UploadSimple,
  DownloadSimple,
  ArrowCounterClockwise,
  CheckCircle,
  Warning,
  ArrowClockwise,
  X,
  Info,
  FolderSimple,
  SlidersHorizontal,
  BookOpen,
  ArrowLeft,
  ArrowRight,
  SquaresFour,
} from "@phosphor-icons/react";
import type {
  Analysis,
  Bootstrap,
  CaseBundle,
  CaseData,
  Completeness,
  Parameters,
  CourseCatalog,
  ControlQuery,
} from "./types";
import { canonical, downloadJson, request } from "./api";
import {
  parseRoute,
  entityHref,
  graphHref,
  labHref,
  caseHref,
  analysisHref,
  reasoningGraphHref,
  settingsHref,
  recordsHref,
  governanceHref,
  analysisPages,
} from "./routes";
import type { Address } from "./routes";
const GraphView = lazy(() => import("./components/GraphView"));
import Inspector from "./components/Inspector";
import ActionMenu from "./components/ActionMenu";
import ConfirmDiscard from "./components/ConfirmDiscard";
import type { Confirmation } from "./components/ConfirmDiscard";
import type { TaskInputUpdate } from "./components/TaskInputs";
import CaseOverview from "./components/CaseOverview";
import CaseEntityDetails from "./components/CaseEntityDetails";
import AnalysisDirectory from "./components/AnalysisDirectory";
import AnalysisSettings from "./components/AnalysisSettings";
import AnalysisRecords from "./components/AnalysisRecords";
const Experiments = lazy(() => import("./components/Experiments"));
const RulesPage = lazy(() => import("./components/RulesPage"));
const Governance = lazy(() => import("./components/Governance"));
const CoursePortal = lazy(() => import("./components/CoursePortal"));
const LabOne = lazy(() => import("./components/LabOne"));
const LabTwo = lazy(() => import("./components/LabTwo"));
const initialParameters: Parameters = {
  method: "hybrid_locked",
  threshold: 0.5,
  model_source: "manual",
};
const fullCompleteness: Completeness = {
  holdings: true,
  trades: true,
  guarantees: true,
  registry: true,
};
function controlClaim(analysis: Analysis, query: ControlQuery) {
  return (
    Object.values(analysis.provenance).find(
      ({ fact }) =>
        fact[0] === "Controls" &&
        fact[1] === query.source &&
        fact[2] === query.target,
    )?.id || null
  );
}
export default function App() {
  const [address, setAddress] = useState<Address>(() =>
    parseRoute(location.hash),
  );
  const route = address.page;
  const addressRef = useRef(address);
  addressRef.current = address;
  const headerRef = useRef<HTMLElement>(null);
  useEffect(() => {
    const header = headerRef.current;
    if (!header) return;
    const root = document.documentElement;
    const updateHeight = () => {
      const bounds = header.getBoundingClientRect();
      root.style.setProperty(
        "--app-header-height",
        Math.ceil(bounds.height) + "px",
      );
      const menu = header.querySelector<HTMLElement>(".research-menu");
      if (menu) {
        header.style.setProperty(
          "--header-menu-offset",
          Math.max(
            8,
            Math.ceil(bounds.bottom - menu.getBoundingClientRect().bottom) + 8,
          ) + "px",
        );
      }
    };
    updateHeight();
    const observer = new ResizeObserver(updateHeight);
    observer.observe(header);
    return () => {
      observer.disconnect();
      root.style.removeProperty("--app-header-height");
    };
  }, []);
  const [governanceVisited, setGovernanceVisited] = useState(
    route === "governance",
  );
  useEffect(() => {
    if (route === "governance") setGovernanceVisited(true);
  }, [route]);
  const [visitedTasks, setVisitedTasks] = useState<string[]>(() => {
    const a = parseRoute(location.hash);
    return a.page === "task" ? [a.labId + ":" + a.moduleId] : [];
  });
  useEffect(() => {
    if (address.page !== "task") return;
    const id = address.labId + ":" + address.moduleId;
    setVisitedTasks((old) => (old.includes(id) ? old : [...old, id]));
  }, [address.page, address.labId, address.moduleId]);
  const [catalog, setCatalog] = useState<CourseCatalog | null>(null);
  const [catalogError, setCatalogError] = useState("");
  const catalogRequest = useRef<AbortController | null>(null);
  const loadCatalog = useCallback(async () => {
    catalogRequest.current?.abort();
    const controller = new AbortController();
    catalogRequest.current = controller;
    setCatalogError("");
    try {
      const value = await request<CourseCatalog>("/api/course", {
        signal: controller.signal,
      });
      if (!controller.signal.aborted) setCatalog(value);
    } catch (e) {
      if (!controller.signal.aborted)
        setCatalogError(e instanceof Error ? e.message : "实验目录载入失败");
    }
  }, []);
  useEffect(() => {
    void loadCatalog();
    return () => catalogRequest.current?.abort();
  }, [loadCatalog]);
  const [boot, setBoot] = useState<Bootstrap | null>(null);
  const [workspace, setWorkspace] = useState<CaseBundle | null>(null);
  const [parameters, setParameters] = useState<Parameters>(initialParameters);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [detailQuery, setDetailQuery] = useState<ControlQuery | null>(null);
  const detailQueryRef = useRef(detailQuery);
  detailQueryRef.current = detailQuery;
  const [selected, setSelected] = useState(
    () => parseRoute(location.hash).entityId || "C3",
  );
  const selectedRef = useRef(selected);
  selectedRef.current = selected;
  const [claim, setClaim] = useState<string | null>(null);
  const [activeFact, setActiveFact] = useState<string | null>(null);
  const [inputValid, setInputValid] = useState(true);
  const [inputEpoch, setInputEpoch] = useState(0);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [errorScope, setErrorScope] = useState<"case" | "analysis">("case");
  const [notice, setNotice] = useState("");
  const [noticeScope, setNoticeScope] = useState<"case" | "analysis">("case");
  const [edited, setEdited] = useState(false);
  const [savedSignature, setSavedSignature] = useState<string | null>(null);
  const [confirmation, setConfirmation] = useState<Confirmation | null>(null);
  const confirmationRef = useRef<((accepted: boolean) => void) | null>(null);
  const busyRef = useRef(true);
  busyRef.current = busy;
  const unsaved =
    !!workspace &&
    canonical({
      case: workspace.case,
      completeness: workspace.completeness,
    }) !== savedSignature;
  function resolveConfirmation(accepted: boolean) {
    const resolve = confirmationRef.current;
    confirmationRef.current = null;
    setConfirmation(null);
    resolve?.(accepted);
  }
  async function canReplace(title: string, action: string) {
    if (!unsaved && inputValid) return true;
    if (confirmationRef.current) return false;
    return new Promise<boolean>((resolve) => {
      confirmationRef.current = resolve;
      setConfirmation({
        title,
        action,
        description:
          "当前案例有尚未导出的修改。继续会替换这些输入；已计算不等于已保存。算法和阈值保持不变。",
      });
    });
  }
  const settingsChanged =
    canonical(parameters) !== canonical(initialParameters);
  useEffect(() => {
    if (!unsaved && inputValid && !settingsChanged) return;
    const warn = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [unsaved, inputValid, settingsChanged]);
  const inputRef = useRef<HTMLInputElement>(null);
  const requestRef = useRef<AbortController | null>(null);
  const bootstrapRef = useRef<AbortController | null>(null);
  const originalRef = useRef<CaseBundle | null>(null);
  const dirty = useMemo(
    () =>
      !!workspace &&
      (!analysis ||
        canonical({
          case: workspace.case,
          completeness: workspace.completeness,
          ...parameters,
        }) !==
          canonical({
            case: analysis.case,
            completeness: analysis.completeness,
            ...analysis.parameters,
          })),
    [workspace, parameters, analysis],
  );
  const acceptCase = useCallback(
    (
      bundle: CaseBundle,
      customized = false,
      keepSelected?: string,
      replacing = true,
    ) => {
      const entities = bundle.case.entities;
      const wanted =
        keepSelected || addressRef.current.entityId || selectedRef.current;
      const target = entities.some((item) => item.eid === wanted)
        ? wanted
        : entities.find((item) => item.kind === "Company")!.eid;
      if (
        replacing &&
        addressRef.current.entityId &&
        !entities.some((item) => item.eid === addressRef.current.entityId)
      ) {
        const current = addressRef.current;
        const nextHash =
          current.page === "graph"
            ? graphHref(target)
            : current.page === "reasoningGraph"
              ? reasoningGraphHref(target)
              : current.page === "caseEntity"
                ? caseHref(target)
                : entityHref(
                    target,
                    current.page === "evidence" || current.page === "edit"
                      ? current.page
                      : "entity",
                  );
        history.replaceState(null, "", nextHash);
        setAddress(parseRoute(nextHash));
      }
      setWorkspace(bundle);
      setAnalysis(null);
      setDetailQuery(null);
      detailQueryRef.current = null;
      setSelected(target);
      setClaim(null);
      setActiveFact(null);
      setEdited(customized);
      setInputValid(true);
      setInputEpoch((value) => value + 1);
      setSavedSignature(
        canonical({ case: bundle.case, completeness: bundle.completeness }),
      );
      originalRef.current = structuredClone(bundle);
      setError("");
      setNotice("");
    },
    [],
  );
  const runAnalysis = useCallback(
    async (
      bundle: CaseBundle,
      params: Parameters,
      keepSelected?: string,
      customized = false,
      replacing = false,
    ) => {
      busyRef.current = true;
      requestRef.current?.abort();
      const controller = new AbortController();
      requestRef.current = controller;
      setBusy(true);
      setErrorScope("analysis");
      setError("");
      setNotice("");
      try {
        const result = await request<Analysis>("/api/analyze", {
          method: "POST",
          body: JSON.stringify({
            case: bundle.case,
            completeness: bundle.completeness,
            ...params,
          }),
          signal: controller.signal,
        });
        if (controller.signal.aborted) return;
        setWorkspace({
          id: bundle.id,
          case: result.case,
          completeness: result.completeness,
        });
        setParameters((current) =>
          canonical(current) === canonical(params)
            ? result.parameters
            : current,
        );
        setAnalysis(result);
        const wanted = replacing
          ? keepSelected || addressRef.current.entityId || selectedRef.current
          : addressRef.current.entityId || selectedRef.current;
        const target =
          wanted && result.entities.some((e) => e.eid === wanted)
            ? wanted
            : result.default_company;
        if (
          replacing &&
          addressRef.current.entityId &&
          !result.entities.some((e) => e.eid === addressRef.current.entityId)
        ) {
          const currentPage = addressRef.current.page;
          const nextHash =
            currentPage === "graph"
              ? graphHref(target)
              : currentPage === "reasoningGraph"
                ? reasoningGraphHref(target)
                : currentPage === "caseEntity"
                  ? caseHref(target)
                  : entityHref(
                      target,
                      currentPage === "evidence" || currentPage === "edit"
                        ? currentPage
                        : "entity",
                    );
          history.replaceState(null, "", nextHash);
          setAddress(parseRoute(nextHash));
        }
        setSelected(target);
        const query = detailQueryRef.current;
        setClaim(
          query &&
            query.target === target &&
            addressRef.current.moduleId === "deterministic"
            ? controlClaim(result, query)
            : result.decisions[target]?.claim_ids[0] || null,
        );
        setActiveFact(null);
        setEdited(customized);
        setInputValid(true);
        if (replacing) setInputEpoch((value) => value + 1);
        if (
          !replacing &&
          !customized &&
          !keepSelected &&
          !addressRef.current.entityId
        ) {
          setNotice("");
          return result;
        }
        setNoticeScope("analysis");
        setNotice(
          result.cached
            ? "已读取相同输入的计算结果"
            : "推理完成，图谱与证据已同步",
        );
        return result;
      } catch (e) {
        if (controller.signal.aborted) return;
        setError(e instanceof Error ? e.message : "推理未完成，请重试。");
        return null;
      } finally {
        if (requestRef.current === controller) {
          busyRef.current = false;
          setBusy(false);
        }
      }
    },
    [],
  );
  const start = useCallback(async () => {
    bootstrapRef.current?.abort();
    const controller = new AbortController();
    bootstrapRef.current = controller;
    busyRef.current = true;
    setBusy(true);
    setErrorScope("case");
    setError("");
    try {
      const metadata = await request<Bootstrap>("/api/bootstrap", {
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      setBoot(metadata);
      const bundle = await request<CaseBundle>(
        "/api/cases/" + metadata.default_case,
        { signal: controller.signal },
      );
      if (controller.signal.aborted) return;
      acceptCase(bundle, false, undefined, false);
      busyRef.current = false;
      setBusy(false);
    } catch (e) {
      if (controller.signal.aborted) return;
      busyRef.current = false;
      setBusy(false);
      setError(e instanceof Error ? e.message : "服务连接失败");
    }
  }, [acceptCase]);
  useEffect(() => {
    void start();
    return () => {
      bootstrapRef.current?.abort();
      requestRef.current?.abort();
      confirmationRef.current?.(false);
      confirmationRef.current = null;
    };
  }, [start]);
  useEffect(() => {
    const change = () => {
      setAddress(parseRoute(location.hash));
      window.scrollTo(0, 0);
      requestAnimationFrame(() =>
        document.getElementById("main-content")?.focus({ preventScroll: true }),
      );
    };
    window.addEventListener("hashchange", change);
    return () => window.removeEventListener("hashchange", change);
  }, []);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 2500);
    return () => clearTimeout(timer);
  }, [notice]);
  useEffect(() => {
    if (
      !workspace ||
      analysis ||
      busy ||
      busyRef.current ||
      error ||
      route === "edit" ||
      route === "settings" ||
      !analysisPages.includes(route)
    )
      return;
    void runAnalysis(workspace, parameters, selectedRef.current, edited);
  }, [
    workspace,
    analysis,
    busy,
    error,
    route,
    parameters,
    edited,
    runAnalysis,
  ]);
  useEffect(() => {
    let nextHash: string | undefined;
    if (location.hash.startsWith("#/entity/") && address.entityId) {
      nextHash = entityHref(
        address.entityId,
        route === "evidence" || route === "edit" ? route : "entity",
      );
    } else if (location.hash === "#/settings") nextHash = settingsHref();
    else if (location.hash === "#/verification") nextHash = recordsHref();
    else if (location.hash === "#/experiments")
      nextHash = labHref("lab02", "optimization") + "/results";
    else if (location.hash === "#/rules")
      nextHash = labHref("lab02", "modeling") + "/rules";
    else if (location.hash === "#/governance") nextHash = governanceHref();
    if (nextHash && route !== "notFound")
      history.replaceState(null, "", nextHash);
  }, [address]);
  const run = useCallback(() => {
    if (workspace && !busyRef.current && inputValid && !confirmationRef.current)
      void runAnalysis(workspace, parameters, selected, edited);
  }, [workspace, parameters, selected, edited, busy, inputValid, runAnalysis]);
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (
        (e.ctrlKey || e.metaKey) &&
        e.key === "Enter" &&
        analysisPages.includes(route)
      ) {
        e.preventDefault();
        run();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [run, route]);
  useEffect(() => {
    if (
      address.entityId &&
      workspace?.case.entities.some((e) => e.eid === address.entityId) &&
      address.entityId !== selected
    ) {
      setSelected(address.entityId);
      setClaim(analysis?.decisions[address.entityId]?.claim_ids[0] || null);
      setActiveFact(null);
    }
  }, [address.entityId, workspace, analysis, selected]);
  function chooseEntity(id: string) {
    if (detailQueryRef.current?.target !== id) {
      setDetailQuery(null);
      detailQueryRef.current = null;
    }
    setSelected(id);
    setClaim(analysis?.decisions[id]?.claim_ids[0] || null);
    setActiveFact(null);
  }
  async function chooseCase(id: string) {
    if (busyRef.current) return false;
    if (id === workspace?.id) return true;
    if (!(await canReplace("切换案例？", "切换案例")) || busyRef.current)
      return false;
    busyRef.current = true;
    setBusy(true);
    setErrorScope("case");
    setError("");
    const controller = new AbortController();
    requestRef.current = controller;
    try {
      const bundle = await request<CaseBundle>("/api/cases/" + id, {
        signal: controller.signal,
      });
      if (controller.signal.aborted) return false;
      acceptCase(bundle);
      busyRef.current = false;
      setBusy(false);
      return true;
    } catch (e) {
      if (!controller.signal.aborted) {
        setError(e instanceof Error ? e.message : "案例加载失败");
        busyRef.current = false;
        setBusy(false);
      }
      return false;
    }
  }
  async function reset() {
    if (!originalRef.current || busyRef.current) return;
    if (
      !(await canReplace("恢复案例初始输入？", "恢复初始输入")) ||
      busyRef.current
    )
      return;
    if (!analysisPages.includes(route)) {
      acceptCase(structuredClone(originalRef.current), false, selected);
      return;
    }
    const accepted = await runAnalysis(
      structuredClone(originalRef.current),
      parameters,
      selected,
      false,
      true,
    );
    if (accepted)
      setSavedSignature(
        canonical({ case: accepted.case, completeness: accepted.completeness }),
      );
  }
  function applyTaskInputs(update: TaskInputUpdate) {
    if (busyRef.current) return;
    setWorkspace((old) => (old ? { ...old, ...update } : old));
    if (update.case) setEdited(true);
    setActiveFact(null);
  }
  const changeParameters = useCallback((update: Partial<Parameters>) => {
    setParameters((old) => ({ ...old, ...update }));
  }, []);
  function prepareTaskDetail(target: string, method?: string, source?: string) {
    if (!workspace || busyRef.current) return;
    const nextParameters = method ? { ...parameters, method } : parameters;
    const query = source ? { source, target } : null;
    setDetailQuery(query);
    detailQueryRef.current = query;
    const matches =
      analysis &&
      canonical({
        case: workspace.case,
        completeness: workspace.completeness,
        ...nextParameters,
      }) ===
        canonical({
          case: analysis.case,
          completeness: analysis.completeness,
          ...analysis.parameters,
        });
    // A task detail must use the task's current input, never an unrelated old snapshot.
    if (!matches) setAnalysis(null);
    setParameters(nextParameters);
    setSelected(target);
    setClaim(
      matches && analysis
        ? query
          ? controlClaim(analysis, query)
          : analysis.decisions[target]?.claim_ids[0] || null
        : null,
    );
    setActiveFact(null);
    setError("");
  }
  function patchCase(update: (old: CaseData) => CaseData) {
    if (busyRef.current) return;
    setWorkspace((old) => (old ? { ...old, case: update(old.case) } : old));
    setEdited(true);
    setActiveFact(null);
  }
  function changeHolding(index: number, value: number) {
    if (
      !workspace ||
      busy ||
      index < 0 ||
      !Number.isFinite(value) ||
      value <= 0
    )
      return;
    const h = workspace.case.holdings[index];
    if (!h) return;
    const other = workspace.case.holdings.reduce(
      (s, r, i) => s + (i !== index && r.target === h.target ? r.ratio : 0),
      0,
    );
    if (value + other > 1 + 1e-9) {
      setError("同一企业的已披露持股不能超过100%。");
      return;
    }
    patchCase((old) => ({
      ...old,
      holdings: old.holdings.map((r, i) =>
        i === index ? { ...r, ratio: value } : r,
      ),
    }));
  }
  function changeOpacity(id: string, value: number) {
    patchCase((old) => ({
      ...old,
      entities: old.entities.map((e) =>
        e.eid === id ? { ...e, opacity: value } : e,
      ),
    }));
  }
  function changeCompleteness(key: keyof Completeness, value: boolean) {
    if (busyRef.current) return;
    setWorkspace((old) =>
      old
        ? { ...old, completeness: { ...old.completeness, [key]: value } }
        : old,
    );
  }
  async function importCase(file: File) {
    if (busyRef.current) return;
    setErrorScope("case");
    if (file.size > (boot?.limits.bytes || 262144)) {
      setError("文件超过网页输入上限，请精简案例后再导入。");
      return;
    }
    try {
      const raw: unknown = JSON.parse(await file.text());
      if (!raw || typeof raw !== "object" || Array.isArray(raw))
        throw new Error("请选择一个完整的图JSON对象。");
      const object = raw as Record<string, unknown>;
      if (
        "case" in object &&
        Object.keys(object).some(
          (k) => !["case", "target_id", "completeness"].includes(k),
        )
      )
        throw new Error(
          "请导入观测图请求，不要包含风险标签、干净图真值或实验审计字段。",
        );
      const value = ("case" in object ? object.case : object) as CaseData;
      const complete = ("case" in object ? object.completeness : undefined) as
        | Completeness
        | undefined;
      const bundle: CaseBundle = {
        id: "custom",
        case: value,
        completeness: complete || fullCompleteness,
      };
      if (!Array.isArray(value?.entities))
        throw new Error("JSON缺少entities数组，可先导出当前案例作为模板。");
      if (
        busyRef.current ||
        !(await canReplace("导入并替换当前案例？", "导入案例")) ||
        busyRef.current
      )
        return;
      busyRef.current = true;
      setBusy(true);
      setError("");
      const controller = new AbortController();
      requestRef.current = controller;
      try {
        const accepted = await request<CaseBundle>("/api/cases/validate", {
          method: "POST",
          signal: controller.signal,
          body: JSON.stringify({
            case: bundle.case,
            completeness: bundle.completeness,
          }),
        });
        if (!controller.signal.aborted)
          acceptCase(
            accepted,
            true,
            typeof object.target_id === "string" ? object.target_id : undefined,
          );
      } finally {
        if (requestRef.current === controller) {
          busyRef.current = false;
          setBusy(false);
        }
      }
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "无法读取该JSON，请检查文件格式。",
      );
    }
  }
  function exportCase() {
    if (workspace) {
      const target =
        workspace.case.entities.find(
          (e) => e.eid === selected && e.kind === "Company",
        )?.eid ||
        workspace.case.entities.find((e) => e.kind === "Company")?.eid;
      downloadJson(
        {
          case: workspace.case,
          target_id: target,
          completeness: workspace.completeness,
        },
        "equity-case.json",
      );
      setSavedSignature(
        canonical({
          case: workspace.case,
          completeness: workspace.completeness,
        }),
      );
      setNoticeScope("case");
      setNotice(
        inputValid
          ? "已发起案例下载，请确认文件已保存"
          : "已导出有效输入；未通过校验的文本未写入文件",
      );
    }
  }
  const entity = workspace?.case.entities.find(
    (e) => e.eid === (address.entityId || selected),
  );
  const missingEntity =
    !!address.entityId &&
    !!workspace &&
    !workspace.case.entities.some((e) => e.eid === address.entityId);
  const isEntityPage = ["entity", "evidence", "edit"].includes(route);
  const isCasePage = ["overview", "caseEntity", "graph"].includes(route);
  const isRawCasePage = isCasePage;
  const isAnalysisPage = analysisPages.includes(route);
  const visibleError = isRawCasePage && errorScope === "analysis" ? "" : error;
  const titles = {
    notFound: ["页面未找到", "这个地址无效或已变更，请返回案例概览继续。"],
    overview: ["案例概览", "浏览共享实体资料与输入关系，不包含实验推理结果。"],
    caseEntity: ["实体资料", "查看当前案例的原始资料与关系记录。"],
    analysis: ["全案例分析", "实验二 · 汇总企业结构证明、评分与复核策略。"],
    reasoningGraph: ["推理关系图", "实验二 · 查看推导控制关系与完整证明过程。"],
    entity: ["企业分析", "当前企业的复核行动、评分与已知线索。"],
    evidence: ["推理证据", "沿着前提与规则，追溯当前结论的来源。"],
    edit: ["调整企业数据", "修改当前案例草稿，重新推理后再查看结论。"],
    graph: ["关系图谱", "查看企业与自然人之间的持股、交易、担保和亲属关系。"],
    settings: ["分析设置", "选择推理方法、模型参数与资料完整性。"],
    verification: [
      "验证与推理记录",
      "反向验证、完整步骤与分析导出，集中在这一页。",
    ],
  };
  useEffect(() => {
    const names: Record<string, string> = {
      home: "选择实验",
      lab: "实验任务目录",
      task: "实验任务",
      governance: "证据与更新",
      experiments: "实验对比",
      rules: "规则知识库",
      ...Object.fromEntries(
        Object.entries(titles).map(([key, value]) => [key, value[0]]),
      ),
    };
    document.title =
      (entity && (isEntityPage || route === "caseEntity")
        ? entity.name + " · "
        : "") +
      (address.page === "task"
        ? catalog?.labs
            .find((l) => l.id === address.labId)
            ?.modules.find((m) => m.id === address.moduleId)?.title ||
          "实验任务"
        : address.page === "lab"
          ? "实验" + (address.labId === "lab01" ? "一" : "二")
          : names[route]) +
      (address.labId === "lab02" ? " · 实验二" : "") +
      " · 股权推理";
  }, [
    route,
    entity?.name,
    isEntityPage,
    address.labId,
    address.moduleId,
    catalog,
  ]);
  const closeCaseMenu = () => {
    const menu = document.getElementById(
      "case-manager",
    ) as HTMLDetailsElement | null;
    if (menu) menu.open = false;
  };
  const openEntity = (id: string) => {
    chooseEntity(id);
    location.hash = entityHref(
      id,
      address.moduleId === "deterministic" ? "evidence" : "entity",
    );
  };
  return (
    <div className="app multipage-app">
      <a
        className="skip-link"
        href="#main-content"
        onClick={(e) => {
          e.preventDefault();
          document.getElementById("main-content")?.focus();
        }}
      >
        跳转到主要内容
      </a>
      <header ref={headerRef} className="app-header course-app-header">
        <a className="brand" href="#/home" aria-label="股权知识实验平台首页">
          <span className="brand-symbol">
            <Graph size={24} weight="bold" />
          </span>
          <span>
            股权知识实验
            <span className="brand-subtitle">知识表示 · 智能推理</span>
          </span>
        </a>
        <nav aria-label="主导航" className="course-navigation">
          <a
            href="#/home"
            className={route === "home" ? "active" : ""}
            aria-current={route === "home" ? "page" : undefined}
          >
            <SquaresFour size={18} />
            <span>选择实验</span>
          </a>
          <a
            href={labHref("lab01")}
            className={address.labId === "lab01" ? "active" : ""}
            aria-current={address.labId === "lab01" ? "location" : undefined}
          >
            <BookOpen size={18} />
            <span>实验一</span>
          </a>
          <a
            href={labHref("lab02")}
            className={address.labId === "lab02" ? "active" : ""}
            aria-current={address.labId === "lab02" ? "location" : undefined}
          >
            <TreeStructure size={18} />
            <span>实验二</span>
          </a>
          <ActionMenu
            label="公共工具"
            icon={<SlidersHorizontal size={18} />}
            className="research-menu"
            active={isCasePage}
          >
            <p className="menu-heading">共享案例 · 原始数据</p>
            <a className="menu-link" href="#/workbench" aria-label="案例概览">
              <Graph size={18} />
              <span>
                <strong>案例概览</strong>
                <small>企业、自然人与原始关系记录</small>
              </span>
            </a>
            <a className="menu-link" href={graphHref()} aria-label="关系图谱">
              <Graph size={18} />
              <span>
                <strong>关系图谱</strong>
                <small>查看共享案例的输入关系</small>
              </span>
            </a>
          </ActionMenu>
        </nav>
        <div className="header-status">
          <span className={"status-dot " + (visibleError ? "warning" : "")} />
          <span>
            {busy
              ? !workspace
                ? "正在载入案例"
                : "本地引擎计算中"
              : visibleError
                ? "操作未完成"
                : boot
                  ? "本地引擎已就绪"
                  : "等待连接引擎"}
          </span>
        </div>
      </header>
      <input
        ref={inputRef}
        type="file"
        className="sr-only"
        accept=".json,application/json"
        aria-label="导入股权图JSON"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) {
            closeCaseMenu();
            void importCase(f);
          }
          e.target.value = "";
        }}
      />
      <ConfirmDiscard
        confirmation={confirmation}
        onResolve={resolveConfirmation}
        onExport={exportCase}
      />
      {governanceVisited && (
        <div hidden={route !== "governance"}>
          <Suspense
            fallback={
              <main
                id={route === "governance" ? "main-content" : undefined}
                tabIndex={-1}
                className="page-shell"
              >
                <div
                  className="skeleton experiment-skeleton"
                  aria-label="正在加载证据与更新"
                />
              </main>
            }
          >
            <Governance
              currentCase={workspace?.case || null}
              active={route === "governance"}
            />
          </Suspense>
        </div>
      )}
      {catalog &&
        visitedTasks.map((id) => {
          const [labId, moduleId] = id.split(":");
          const lab = catalog.labs.find((l) => l.id === labId);
          if (!lab) return null;
          const active =
            address.page === "task" &&
            address.labId === labId &&
            address.moduleId === moduleId;
          return (
            <div key={id} hidden={!active}>
              <Suspense
                fallback={
                  active ? (
                    <main
                      id="main-content"
                      tabIndex={-1}
                      className="page-shell course-shell"
                    >
                      <div
                        className="skeleton course-skeleton"
                        aria-label="正在载入实验任务"
                      />
                    </main>
                  ) : null
                }
              >
                {labId === "lab01" ? (
                  <LabOne lab={lab} moduleId={moduleId} active={active} />
                ) : (
                  <LabTwo
                    lab={lab}
                    moduleId={moduleId}
                    active={active}
                    workspace={workspace}
                    boot={boot}
                    parameters={parameters}
                    workspaceBusy={busy}
                    workspaceError={error}
                    onReconnect={() => void start()}
                    onChooseCase={chooseCase}
                    onApplyInputs={applyTaskInputs}
                    onParameters={changeParameters}
                    onOpenDetails={prepareTaskDetail}
                  />
                )}
              </Suspense>
            </div>
          );
        })}
      {route === "governance" ? null : ["home", "lab", "task"].includes(
          route,
        ) ? (
        <Suspense
          fallback={
            <main
              id="main-content"
              tabIndex={-1}
              className="page-shell course-shell"
            >
              <div
                className="skeleton course-skeleton"
                aria-label="正在加载实验页面"
              />
            </main>
          }
        >
          {route !== "task" || !catalog ? (
            <CoursePortal
              catalog={catalog}
              labId={address.labId}
              error={catalogError}
              onRetry={() => void loadCatalog()}
            />
          ) : !visitedTasks.includes(address.labId + ":" + address.moduleId) ? (
            <main
              id="main-content"
              tabIndex={-1}
              className="page-shell course-shell"
            >
              <div className="skeleton course-skeleton" />
            </main>
          ) : null}
        </Suspense>
      ) : route === "experiments" || route === "rules" ? (
        <Suspense
          fallback={
            <main id="main-content" tabIndex={-1} className="page-shell">
              <div className="skeleton experiment-skeleton" />
            </main>
          }
        >
          {route === "experiments" ? (
            <Experiments />
          ) : (
            <RulesPage caseData={workspace?.case ?? null} />
          )}
        </Suspense>
      ) : (
        <main
          id="main-content"
          tabIndex={-1}
          className={"page-shell focused-page " + route + "-page"}
        >
          {route !== "overview" && (
            <nav className="page-breadcrumb" aria-label="当前位置">
              <a
                href={
                  address.labId === "lab02" ? labHref("lab02") : "#/workbench"
                }
              >
                <ArrowLeft size={15} />
                {address.labId === "lab02" ? "实验二" : "返回案例概览"}
              </a>
              {address.labId === "lab02" && (
                <>
                  <span>/</span>
                  <a href={labHref("lab02", address.moduleId)}>
                    {catalog?.labs
                      .find((l) => l.id === "lab02")
                      ?.modules.find((m) => m.id === address.moduleId)?.title ||
                      "实验任务"}
                  </a>
                </>
              )}
              <span>/</span>
              <span>
                {isEntityPage || route === "caseEntity"
                  ? entity?.name || address.entityId
                  : titles[route as keyof typeof titles][0]}
              </span>
            </nav>
          )}
          <header className="page-header focused-header">
            <div>
              <h1>{titles[route as keyof typeof titles][0]}</h1>
              <p>{titles[route as keyof typeof titles][1]}</p>
            </div>
            <div className="page-actions">
              <ActionMenu
                label="案例管理"
                icon={<FolderSimple size={18} />}
                id="case-manager"
                className="case-menu"
              >
                <p className="menu-heading">切换与管理案例</p>
                <label className="menu-field">
                  选择案例
                  <select
                    aria-label="选择案例"
                    value={workspace?.id || "base"}
                    onChange={(e) => {
                      closeCaseMenu();
                      void chooseCase(e.target.value);
                    }}
                    disabled={busy || !boot}
                  >
                    {boot?.cases.map((c) => (
                      <option value={c.id} key={c.id}>
                        {c.name}
                      </option>
                    ))}
                    {workspace?.id === "custom" && (
                      <option value="custom">导入的案例</option>
                    )}
                    {!boot && <option>正在连接…</option>}
                  </select>
                </label>
                <div className="menu-divider" />
                <button
                  className="menu-action"
                  aria-label="导入案例"
                  disabled={busy}
                  onClick={() => inputRef.current?.click()}
                >
                  <UploadSimple size={18} />
                  <span>
                    导入案例<small>使用完整的股权图 JSON</small>
                  </span>
                </button>
                <button
                  className="menu-action"
                  aria-label="导出案例"
                  disabled={!workspace || busy}
                  onClick={() => {
                    exportCase();
                    closeCaseMenu();
                  }}
                >
                  <DownloadSimple size={18} />
                  <span>
                    导出案例<small>保存当前草稿与完整性设置</small>
                  </span>
                </button>
                <button
                  className="menu-action"
                  aria-label="重置案例"
                  disabled={!workspace || busy}
                  onClick={() => {
                    closeCaseMenu();
                    void reset();
                  }}
                >
                  <ArrowCounterClockwise size={18} />
                  <span>
                    重置案例<small>回到当前案例的初始输入</small>
                  </span>
                </button>
              </ActionMenu>
              {isRawCasePage ? (
                <a
                  className="button primary"
                  href={graphHref(
                    route === "caseEntity" ? address.entityId : undefined,
                  )}
                >
                  <Graph size={18} />
                  查看关系图谱
                </a>
              ) : (
                <button
                  className="button primary run-button"
                  onClick={run}
                  disabled={!workspace || busy || !inputValid || !!confirmation}
                >
                  <Play size={17} weight="fill" />
                  {busy ? "正在推理…" : "运行推理"}
                  <kbd>Ctrl ↵</kbd>
                </button>
              )}
            </div>
          </header>
          {visibleError && (
            <div className="error-banner" role="alert">
              <Warning size={21} />
              <div>
                <strong>操作未完成</strong>
                <p>{visibleError}</p>
              </div>
              <button
                className="icon-button"
                aria-label="关闭错误提示"
                onClick={() => setError("")}
              >
                <X size={18} />
              </button>
            </div>
          )}
          {workspace && (
            <div className="workspace-context focused-context">
              <span>
                {workspace.case.label}
                {edited && <span className="edited-tag">已调整</span>}
              </span>
              <div>
                {unsaved && <span className="draft-state">草稿未导出</span>}
                {isRawCasePage ? (
                  <span>
                    <CheckCircle size={15} />
                    {busy ? "正在载入数据" : "当前输入数据"}
                  </span>
                ) : (
                  <>
                    {busy ? (
                      <span className="pending-state">
                        <Info size={15} />
                        {analysis
                          ? "正在计算，暂保留上次结果"
                          : "正在计算实验二分析"}
                      </span>
                    ) : dirty ? (
                      <span className="pending-state">
                        <Info size={15} />
                        {analysis ? "存在未应用更改" : "尚未运行实验二分析"}
                      </span>
                    ) : analysis ? (
                      <span>
                        <CheckCircle size={15} />
                        实验二结果 · {analysis.stats.cycles}轮推理 ·{" "}
                        {analysis.stats.derived_facts}条派生事实
                      </span>
                    ) : null}
                    {analysis && !dirty && !busy && (
                      <span className="engine-time">
                        引擎{analysis.stats.elapsed_ms.toFixed(1)}ms
                      </span>
                    )}
                  </>
                )}
              </div>
            </div>
          )}
          {((dirty && !!analysis) || !inputValid) &&
            workspace &&
            isAnalysisPage && (
              <div className="workspace-change" role="status">
                <Info size={20} />
                <div>
                  <strong>
                    {inputValid
                      ? "输入已调整，结果尚未更新"
                      : "持股输入尚未通过校验"}
                  </strong>
                  <p>
                    {inputValid
                      ? "下方分析仍是上次快照。运行推理后才会更新结论；页面跳转保留有效修改。"
                      : "请先修正调整数据页的输入，再运行推理；离开编辑页会丢弃无效文本。"}
                  </p>
                </div>
                {inputValid ? (
                  <button
                    className="button secondary"
                    onClick={run}
                    disabled={busy}
                  >
                    应用并推理
                    <ArrowRight size={16} />
                  </button>
                ) : (
                  <a
                    className="button secondary"
                    href={entityHref(selected, "edit")}
                  >
                    修正输入
                  </a>
                )}
              </div>
            )}
          {detailQuery &&
            address.moduleId === "deterministic" &&
            selected === detailQuery.target &&
            analysis &&
            !dirty &&
            !busy && (
              <div className="detail-query-context" role="status">
                <strong>
                  任务查询：
                  <code>
                    Controls({detailQuery.source}, {detailQuery.target})
                  </code>
                </strong>
                <p>
                  {controlClaim(analysis, detailQuery)
                    ? "已定位到这条控制结论的证明，可继续查看图谱或验证记录。"
                    : "当前输入下未证明这条控制关系。下方仍可查看该企业的其他已知结构结论。"}
                </p>
              </div>
            )}
          {route === "notFound" ? (
            <section className="simple-panel not-found">
              <Graph size={36} />
              <h2>找不到这个页面</h2>
              <p>请检查地址。当前案例和已接受的修改仍然保留。</p>
              <a className="button primary" href="#/workbench">
                返回案例概览
                <ArrowRight size={17} />
              </a>
            </section>
          ) : workspace ? (
            missingEntity ? (
              <section className="simple-panel not-found">
                <h2>当前案例没有这个实体</h2>
                <p>
                  链接指向{address.entityId}，请选择当前案例中的企业或自然人。
                </p>
                <a className="button secondary" href="#/workbench">
                  返回案例概览
                </a>
              </section>
            ) : (
              <>
                {route === "overview" && <CaseOverview data={workspace.case} />}
                {route === "analysis" &&
                  (analysis ? (
                    <AnalysisDirectory
                      data={workspace.case}
                      analysis={analysis}
                      dirty={dirty}
                    />
                  ) : (
                    <section
                      className="simple-panel"
                      aria-label="实验二分析状态"
                    >
                      {busy ? (
                        <>
                          <div
                            className="skeleton experiment-skeleton"
                            aria-label="正在运行实验二分析"
                          />
                          <p>正在读取结构证明、评分与复核策略…</p>
                        </>
                      ) : (
                        <>
                          <h2>尚无实验二分析结果</h2>
                          <p>案例数据已经载入，运行分析后查看各企业的结果。</p>
                          <button
                            className="button primary"
                            onClick={run}
                            disabled={!inputValid}
                          >
                            运行实验二分析
                            <ArrowRight size={16} />
                          </button>
                        </>
                      )}
                    </section>
                  ))}
                {route === "caseEntity" && entity && (
                  <>
                    <div className="entity-page-tools">
                      <label>
                        切换实体
                        <select
                          aria-label="选择实体"
                          value={entity.eid}
                          onChange={(event) => {
                            chooseEntity(event.target.value);
                            location.hash = caseHref(event.target.value);
                          }}
                        >
                          {workspace.case.entities.map((item) => (
                            <option key={item.eid} value={item.eid}>
                              {item.eid} · {item.name}
                            </option>
                          ))}
                        </select>
                      </label>
                    </div>
                    <CaseEntityDetails
                      data={workspace.case}
                      entityId={entity.eid}
                    />
                  </>
                )}
                {isEntityPage && entity && (
                  <>
                    <div className="entity-page-tools">
                      <label>
                        切换实体
                        <select
                          aria-label="选择实体"
                          value={entity.eid}
                          onChange={(e) => {
                            chooseEntity(e.target.value);
                            location.hash = entityHref(
                              e.target.value,
                              route === "evidence"
                                ? "evidence"
                                : route === "edit"
                                  ? "edit"
                                  : "entity",
                            );
                          }}
                        >
                          {workspace.case.entities.map((e) => (
                            <option key={e.eid} value={e.eid}>
                              {e.eid} · {e.name}
                            </option>
                          ))}
                        </select>
                      </label>
                      {route !== "evidence" && (
                        <a
                          href={
                            route === "edit"
                              ? graphHref(entity.eid)
                              : reasoningGraphHref(entity.eid)
                          }
                        >
                          <Graph size={17} />
                          {route === "edit"
                            ? "查看原始关系"
                            : "查看结构关系与证明"}
                          <ArrowRight size={15} />
                        </a>
                      )}
                    </div>
                    <Inspector
                      key={entity.eid + ":" + inputEpoch}
                      data={workspace.case}
                      analysis={analysis}
                      selected={entity.eid}
                      dirty={dirty}
                      busy={busy}
                      claim={claim}
                      activeFact={activeFact}
                      onClaim={setClaim}
                      onActiveFact={setActiveFact}
                      onSelect={openEntity}
                      onHolding={changeHolding}
                      onOpacity={changeOpacity}
                      onRun={run}
                      inputValid={inputValid}
                      onValidity={setInputValid}
                      view={
                        route === "evidence"
                          ? "evidence"
                          : route === "edit"
                            ? "tune"
                            : "result"
                      }
                      standalone
                    />
                    <nav className="related-pages" aria-label="相关页面">
                      {route === "evidence" ? (
                        <a href={labHref("lab02", "deterministic")}>
                          返回确定性推理
                          <ArrowRight size={16} />
                        </a>
                      ) : route === "edit" ? (
                        <a href={labHref("lab02", "modeling")}>
                          返回知识表示与建模
                          <ArrowRight size={16} />
                        </a>
                      ) : (
                        <>
                          <a href={analysisHref()}>
                            返回实验二全案例分析
                            <ArrowRight size={16} />
                          </a>
                          <a href={settingsHref()}>
                            <SlidersHorizontal size={17} />
                            分析设置
                            <ArrowRight size={15} />
                          </a>
                        </>
                      )}
                      {route === "evidence" && (
                        <a href={recordsHref()}>
                          <TreeStructure size={17} />
                          验证与推理记录
                          <ArrowRight size={15} />
                        </a>
                      )}
                      {route === "entity" && (
                        <a href={entityHref(selected, "evidence")}>
                          查看确定性推理证据
                          <ArrowRight size={16} />
                        </a>
                      )}
                    </nav>
                  </>
                )}
                {(route === "graph" || route === "reasoningGraph") && (
                  <>
                    <div className="graph-page-toolbar">
                      <label>
                        定位实体
                        <select
                          aria-label="选择实体"
                          value={selected}
                          onChange={(e) => {
                            chooseEntity(e.target.value);
                            if (address.entityId)
                              location.hash =
                                route === "graph"
                                  ? graphHref(e.target.value)
                                  : reasoningGraphHref(e.target.value);
                          }}
                        >
                          {workspace.case.entities.map((e) => (
                            <option key={e.eid} value={e.eid}>
                              {e.eid} · {e.name}
                            </option>
                          ))}
                        </select>
                      </label>
                      <a
                        className="button secondary"
                        href={
                          route === "graph"
                            ? caseHref(selected)
                            : entityHref(selected, "evidence")
                        }
                      >
                        {route === "graph"
                          ? "查看所选实体资料"
                          : "查看所选实体推理证据"}
                        <ArrowRight size={16} />
                      </a>
                    </div>
                    <div className="standalone-graph">
                      <Suspense
                        fallback={
                          <div className="graph-loading" role="status">
                            <div className="skeleton" />
                            正在加载关系图谱…
                          </div>
                        }
                      >
                        <GraphView
                          key={
                            route + (address.entityId ? ":focused" : ":full")
                          }
                          mode={route === "graph" ? "source" : "reasoning"}
                          data={workspace.case}
                          analysis={route === "graph" ? null : analysis}
                          selected={selected}
                          onSelect={(id) => {
                            chooseEntity(id);
                            if (address.entityId)
                              location.hash =
                                route === "graph"
                                  ? graphHref(id)
                                  : reasoningGraphHref(id);
                          }}
                          claim={route === "graph" ? null : claim}
                          activeFact={route === "graph" ? null : activeFact}
                          onActiveFact={setActiveFact}
                          dirty={route === "graph" ? edited : dirty}
                          busy={busy}
                          initialFocus={!!address.entityId}
                          initialActiveFact={activeFact}
                        />
                      </Suspense>
                    </div>
                    <p className="page-footnote">
                      矩形是企业，圆形是自然人；线上的比例或金额来自当前案例。全图与当前企业关系只是显示范围，不会更改输入。
                    </p>
                  </>
                )}
                {route === "settings" && (
                  <>
                    <AnalysisSettings
                      boot={boot}
                      parameters={parameters}
                      completeness={workspace.completeness}
                      busy={busy}
                      onParameters={changeParameters}
                      onCompleteness={changeCompleteness}
                    />
                    <nav className="related-pages">
                      <a href={entityHref(selected)}>
                        返回企业分析
                        <ArrowRight size={16} />
                      </a>
                      <a href={labHref("lab02", "hybrid")}>
                        返回混合推理任务
                        <ArrowRight size={16} />
                      </a>
                    </nav>
                  </>
                )}
                {route === "verification" && (
                  <AnalysisRecords
                    data={workspace.case}
                    analysis={analysis}
                    dirty={dirty || !inputValid || busy}
                    initialQuery={detailQuery}
                  />
                )}
              </>
            )
          ) : busy ? (
            <div className="simple-loading">
              <div className="skeleton" />
              <span>正在载入共享案例…</span>
            </div>
          ) : (
            <div className="empty-state large">
              <Graph size={42} />
              <h2>连接推理引擎后开始</h2>
              <p>确认本地服务运行后，可以重新连接。</p>
              <button className="button primary" onClick={() => void start()}>
                <ArrowClockwise size={18} />
                重新连接
              </button>
            </div>
          )}
        </main>
      )}
      {notice && (noticeScope === "case" || isAnalysisPage) && (
        <div className="toast" role="status">
          <CheckCircle size={18} />
          {notice}
          <button aria-label="关闭提示" onClick={() => setNotice("")}>
            <X size={15} />
          </button>
        </div>
      )}
    </div>
  );
}
