import { useState } from "react";
import {
  ArrowRight,
  DownloadSimple,
  Play,
  FileText,
  WarningCircle,
} from "@phosphor-icons/react";
import type { CourseLab } from "../types";

type Material = NonNullable<CourseLab["files"]>[number];
const metadata: Record<string, { description: string; format: string }> = {
  "current-report-pdf": {
    description: "9页独立阅读版本：范式、公式、训练、全部实测与负结果。",
    format: "PDF",
  },
  "current-report-word": {
    description: "与PDF相同内容，可继续编辑，不覆盖v1原报告。",
    format: "Word",
  },
  "current-slides": {
    description: "12页可编辑幻灯片，数值来自已核验发布记录。",
    format: "PPTX",
  },
  "current-demo-video": {
    description:
      "约80秒真实浏览器操作，包含4次现场推理。无音轨，提供中文字幕。",
    format: "MP4 · 实际操作",
  },
  "current-presentation-video": {
    description: "约6分钟以内，12页PPT配本机标准合成语音讲解，非真人录音。",
    format: "MP4 · 合成语音",
  },
};
function size(bytes: number) {
  return bytes >= 1_048_576
    ? `${(bytes / 1_048_576).toFixed(1)} MB`
    : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}
function safeFile(value: unknown): value is Material {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false;
  const file = value as Partial<Material>;
  return (
    file &&
    typeof file.id === "string" &&
    typeof file.label === "string" &&
    typeof file.available === "boolean" &&
    typeof file.kind === "string" &&
    ["report", "slides", "video", "captions", "preview"].includes(file.kind) &&
    typeof file.bytes === "number" &&
    Number.isFinite(file.bytes) &&
    file.bytes >= 0 &&
    file.href === `/api/course/files/lab04/${file.id}` &&
    /^[a-z0-9-]+$/.test(file.id)
  );
}

export default function Materials({ lab }: { lab: CourseLab }) {
  const files = (Array.isArray(lab.files) ? lab.files : []).filter(safeFile);
  const current = files.filter(
    (f) =>
      f.id.startsWith("current-") &&
      ["report", "slides", "video"].includes(f.kind),
  );
  const history = files.filter(
    (f) => !f.id.startsWith("current-") && !f.id.startsWith("supplement"),
  );
  const supplements = files.filter((f) => f.id.startsWith("supplement"));
  const videos = current.filter((f) => f.kind === "video" && f.available);
  const [chosen, setChosen] = useState("current-demo-video");
  const selected = videos.find((f) => f.id === chosen) || videos[0];
  const caption =
    selected &&
    files.find(
      (f) =>
        f.id ===
          (selected.id === "current-demo-video"
            ? "current-demo-captions"
            : "current-presentation-captions") && f.available,
    );
  const poster =
    selected &&
    files.find(
      (f) =>
        f.id ===
          (selected.id === "current-demo-video"
            ? "current-demo-poster"
            : "current-presentation-poster") && f.available,
    );
  const [videoError, setVideoError] = useState(false);
  function list(items: Material[]) {
    return (
      <ul className="lab4-material-list">
        {items.map((file) => (
          <li key={file.id}>
            <div className="lab4-material-kind" aria-hidden="true">
              {file.kind === "video" ? (
                <Play size={20} />
              ) : (
                <FileText size={20} />
              )}
            </div>
            <div>
              <h3>{file.label}</h3>
              <p>
                {metadata[file.id]?.description ||
                  (file.id.startsWith("supplement")
                    ? "后续阶段记录，保留各版本的真实结论与边界。"
                    : "原始v1材料；不是当前模型和页面的全部结论。")}
              </p>
              <small>
                {metadata[file.id]?.format || file.kind}
                {file.available ? ` · ${size(file.bytes)}` : ""}
              </small>
            </div>
            {file.available ? (
              <a
                className="button secondary"
                href={file.href}
                aria-label={`下载${file.label}`}
              >
                <DownloadSimple size={15} />
                下载
              </a>
            ) : (
              <span className="lab4-material-missing">当前目录未附带</span>
            )}
          </li>
        ))}
      </ul>
    );
  }
  return (
    <div className="lab4-materials-page">
      <header className="lab4-evaluation-header">
        <div>
          <h2>阅读、讲解与演示，对应同一版证据</h2>
          <p>
            优先使用当前完整材料。所有正式数值来自已发布模型，训练中的候选不会写进结果；v1与补充资料保留在下方。
          </p>
        </div>
      </header>
      <section aria-labelledby="lab4-current-materials">
        <h2 id="lab4-current-materials">当前完整版本</h2>
        {list(current)}
        {!current.length && (
          <p>
            当前目录没有完整材料清单。请从根目录deliverables/lab4/当前版本核对，不能据此声称已提交。
          </p>
        )}
      </section>
      {selected && (
        <section
          className="lab4-video-section"
          aria-labelledby="lab4-video-title"
        >
          <header>
            <h2 id="lab4-video-title">先看实际效果</h2>
            <p>浏览器原生播放器；不会自动播放或预先下载整段视频。</p>
          </header>
          <div
            className="lab4-video-tabs"
            role="group"
            aria-label="选择演示视频"
          >
            {videos.map((file) => (
              <button
                key={file.id}
                type="button"
                aria-pressed={selected.id === file.id}
                onClick={() => {
                  setChosen(file.id);
                  setVideoError(false);
                }}
              >
                {file.id === "current-demo-video" ? "真实操作 Demo" : "PPT讲解"}
              </button>
            ))}
          </div>
          {videoError && (
            <p className="error-banner" role="alert">
              <WarningCircle size={18} />
              视频加载失败，请下载后使用本地播放器打开。
            </p>
          )}
          <video
            key={selected.id}
            controls
            poster={poster?.href}
            preload="none"
            className="lab4-material-video"
            aria-label={selected.label}
            onError={() => setVideoError(true)}
          >
            <source src={selected.href} type="video/mp4" />
            {caption && (
              <track
                kind="captions"
                src={caption.href}
                srcLang="zh"
                label="中文"
                default
              />
            )}
            当前浏览器不支持嵌入视频，请使用上方下载入口。
          </video>
          <p className="lab4-footnote">
            {metadata[selected.id]?.description}
            {caption && (
              <>
                {" "}
                <a href={caption.href}>下载WebVTT字幕</a>
              </>
            )}
          </p>
        </section>
      )}
      {!videos.length && (
        <p className="lab4-materials-notice">
          <WarningCircle size={18} />
          当前源码目录没有附带视频，模型与推理仍可正常使用；请将根目录deliverables随材料另行提供。
        </p>
      )}
      <details className="lab4-material-history">
        <summary>历次补充记录 · 不覆盖负结果</summary>
        {list(supplements)}
      </details>
      <details className="lab4-material-history">
        <summary>v1原始材料 · 历史保留</summary>
        {list(history)}
      </details>
      <p className="lab4-next-page">
        <a href="#/labs/lab04/requirements">
          回到课程要求与证据核对 <ArrowRight size={15} />
        </a>
      </p>
    </div>
  );
}
