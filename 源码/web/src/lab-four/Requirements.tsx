import { ArrowRight, CheckCircle, WarningCircle } from "@phosphor-icons/react";
import type { CourseLab } from "../types";
import type { LabFourSnapshot } from "./types";

export default function Requirements({
  snapshot,
  lab,
}: {
  snapshot: LabFourSnapshot;
  lab: CourseLab;
}) {
  const compared = ["baseline", "maml", "full"].every((method) =>
    [1, 5].every((shot) =>
      snapshot.comparisons.some((r) => r.method === method && r.shot === shot),
    ),
  );
  const verified = !!snapshot.verification?.verified;
  const items = [
    {
      title: "梳理小样本范式，分析特征漂移",
      ready: verified && compared,
      detail:
        "保留度量分类、二阶MAML、特征统计校准的对照；模型推理与特征投影分开，不用二维点位移动伪造适应。",
      path: "experiments/lab04/models.py · calibration.py",
      page: "protocol",
      link: "方法与实验协议",
    },
    {
      title: "轻量ResNet12与训练目标",
      ready:
        verified &&
        snapshot.model.channels.length === 4 &&
        snapshot.training.length > 0,
      detail: `当前轻量骨干通道${snapshot.model.channels.join(" / ")}，${snapshot.model.parameters.toLocaleString("zh-CN")}参数。Baseline使用交叉熵，校准分支加入分布对齐正则；较宽研究另行披露额外预算。`,
      path: "experiments/lab04/models.py · experiment.py · outputs/lab04/training.json",
      page: "protocol",
      link: "结构、损失与训练记录",
    },
    {
      title: "均值／方差校准与原型自适应",
      ready:
        verified &&
        ["mean_only", "variance_only", "prototype_only", "full"].every((m) =>
          snapshot.ablations.some((r) => r.method === m),
        ),
      detail:
        "校准先验来自基础训练类；原型更新只使用支持标签和无标签查询特征。并非每个模块都有正收益，关闭的模块不计为贡献。",
      path: "experiments/lab04/calibration.py · feature_space.py",
      page: "ablations",
      link: "逐模块查看消融",
    },
    {
      title: "mini-ImageNet上的公平比较",
      ready:
        verified &&
        compared &&
        snapshot.dataset.train_classes === 64 &&
        snapshot.dataset.val_classes === 16 &&
        snapshot.dataset.test_classes === 20,
      detail:
        "64/16/20类隔离；5-way 1/5-shot，每类15张query；支持与查询互斥。所有方法使用相同任务与图片，报告配对差与不确定性。",
      path: "outputs/lab04/evaluation_plan.json · episodes.json · results.json",
      page: "evaluation",
      link: "查看核验后的性能",
    },
    {
      title: "消融、局限与后续方向",
      ready:
        verified &&
        snapshot.ablations.length > 0 &&
        snapshot.limitations.length > 0,
      detail:
        "保留负结果与旧版本。区间仅描述固定模型的任务抽样，不是多训练种子或跨域证明；重复使用的验证类别存在选择偏差。",
      path: "outputs/lab04/history/v1/ · experiments/lab04/QUALITY_PLAN.md",
      page: "history",
      link: "历史记录与研究边界",
    },
    {
      title: "本地端侧部署与真实预测",
      ready: snapshot.available && !!snapshot.edge_benchmark,
      detail:
        "已实现Windows本机CPU编码、适应与分类，携带轻量权重可独立运行。这里的“端侧”是本地CPU，不冒称手机、ARM、医疗或工业数据实测。",
      path: "experiments/lab04/inference.py · benchmark_edge.py · 打开网站.cmd",
      page: "deployment",
      link: "查看统一口径的延迟",
    },
  ];
  const files = lab.files || [];
  return (
    <div className="lab4-requirements">
      <header className="lab4-evaluation-header">
        <div>
          <h2>从课程要求，到可检查的证据</h2>
          <p>
            依据你提供的实验四要求核对。代码与实验事实、适用边界、平台提交动作分开呈现，不用一串绿色勾选替代验收。
          </p>
        </div>
      </header>
      <ol className="lab4-requirement-list">
        {items.map((item, i) => (
          <li key={item.title}>
            <span className="lab4-requirement-number">
              {String(i + 1).padStart(2, "0")}
            </span>
            <div>
              <h3>{item.title}</h3>
              <p>{item.detail}</p>
              <code>{item.path}</code>
              <a href={`#/labs/lab04/${item.page}`}>
                {item.link}
                <ArrowRight size={15} />
              </a>
            </div>
            <span
              className={`lab4-requirement-state ${item.ready ? "is-ready" : "is-pending"}`}
            >
              {item.ready ? (
                <CheckCircle size={17} />
              ) : (
                <WarningCircle size={17} />
              )}{" "}
              {item.ready ? "证据已具备" : "待核对记录"}
            </span>
          </li>
        ))}
      </ol>
      <section className="lab4-requirement-boundary">
        <h2>两件不能由网站代替确认的事</h2>
        <p>
          <strong>课程部署口径：</strong>
          已完成本机CPU端侧示范；如果教师另行指定ARM、手机或开发板，需要补相应硬件实测。不能将本机耗时冒充指定设备成绩。
        </p>
        <p>
          <strong>平台提交动作：</strong>
          1–5人组队、关联组员、创建项目及管理员提交需在课程平台由你核对。当前代码无法证明这些外部动作已完成。
        </p>
        <p>
          医疗影像、工业缺陷和小众物种是动机与潜在应用；本实验仅使用mini-ImageNet，未开展上述领域临床或生产验证。
        </p>
      </section>
      <section className="lab4-materials-link">
        <h2>报告、PPT与演示已单独整理</h2>
        <p>
          当前完整版本与v1历史材料分组呈现，不混用旧报告和新分数。当前目录可用材料
          {
            files.filter(
              (file) =>
                file.available &&
                ["report", "slides", "video"].includes(file.kind),
            ).length
          }
          份；独立源码包不一定附带根目录材料。
        </p>
        <a className="button secondary" href="#/labs/lab04/materials">
          打开报告与演示 <ArrowRight size={16} />
        </a>
      </section>
      <section className="lab4-reproduce-commands">
        <h2>核验与复现入口</h2>
        <p>
          以下命令在“源码”目录运行；核验不会重新训练。训练计划及依赖见experiments/lab04/README.md。
        </p>
        <pre>
          <code>{`python -B -X utf8 main.py lab04 verify
python -B -X utf8 main.py lab04 demo --method calibrated --calibration continued --shots 1
python -B -X utf8 核查提交文件.py`}</code>
        </pre>
      </section>
    </div>
  );
}
