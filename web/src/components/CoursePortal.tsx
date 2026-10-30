import { ArrowRight, ArrowLeft, Graph } from "@phosphor-icons/react";
import type { CourseCatalog, LabId } from "../types";
import { labHref } from "../routes";

export default function CoursePortal({
  catalog,
  labId,
  error,
  onRetry,
}: {
  catalog: CourseCatalog | null;
  labId?: LabId;
  error: string;
  onRetry: () => void;
}) {
  const lab = catalog?.labs.find((item) => item.id === labId);
  return (
    <main id="main-content" tabIndex={-1} className="page-shell course-shell">
      {labId && (
        <nav className="course-breadcrumb" aria-label="当前位置">
          <a href="#/home">
            <ArrowLeft size={16} />
            选择实验
          </a>
          <span>/</span>
          <span>{"实验" + (lab?.number || "")}</span>
        </nav>
      )}
      <header className="course-header">
        <div>
          <h1>
            {lab
              ? "实验" + lab.number + " · " + lab.name
              : "股权知识与智能推理实验平台"}
          </h1>
          <p>
            {lab
              ? lab.summary
              : "统一场景：" +
                (catalog?.scene || "企业股权关系、结构线索与人工复核") +
                "。"}
          </p>
        </div>
      </header>
      {error && (
        <div className="error-banner" role="alert">
          <span>{error}</span>
          <button className="button secondary" onClick={onRetry}>
            重新加载
          </button>
        </div>
      )}
      {!catalog && !error ? (
        <div
          className="skeleton course-skeleton"
          aria-label="正在读取实验目录"
        />
      ) : lab ? (
        <section className="module-list" aria-label="实验任务入口">
          {lab.modules.map((module, index) => (
            <a
              key={module.id}
              href={labHref(lab.id, module.id)}
              className="module-row"
              aria-label={module.title}
            >
              <span className="module-number">
                {String(index + 1).padStart(2, "0")}
              </span>
              <div>
                <span className="requirement-label">{module.kind}</span>
                <h2>{module.title}</h2>
                <p>{module.description}</p>
                <small>{module.requirement}</small>
              </div>
              <span className="module-open">
                进入任务
                <ArrowRight size={19} />
              </span>
            </a>
          ))}
        </section>
      ) : catalog ? (
        <>
          <section className="lab-directory" aria-label="选择实验">
            {catalog.labs.map((item) => (
              <a
                className="lab-entry"
                href={labHref(item.id)}
                key={item.id}
                aria-label={"进入实验" + item.number}
              >
                <div className="lab-entry-heading">
                  <span className="lab-number">实验 {item.number}</span>
                  <span>{item.modules.length} 个任务</span>
                </div>
                <h2>{item.name}</h2>
                <p>{item.summary}</p>
                <div className="lab-task-preview">
                  {item.modules.map((module) => (
                    <span key={module.id}>{module.title}</span>
                  ))}
                </div>
                <div className="lab-entry-bottom">
                  <span>
                    进入实验
                    <ArrowRight size={20} />
                  </span>
                </div>
              </a>
            ))}
          </section>
          <nav className="course-utilities" aria-label="公共入口">
            <a href="#/workbench">
              <Graph size={18} />
              共享案例与关系图谱
              <ArrowRight size={16} />
            </a>
          </nav>
        </>
      ) : null}
    </main>
  );
}
