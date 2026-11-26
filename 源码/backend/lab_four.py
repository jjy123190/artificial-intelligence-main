"""Adapt verified experiment files to the course website without eager ML imports."""
from __future__ import annotations

import base64
import io
import hashlib
import json
import math
import threading
from pathlib import Path

from .service import WebError


METHODS = [
    {"id": "baseline", "name": "Baseline", "description": "交叉熵训练ResNet12，用支持样本的平均特征建立最近原型分类器。"},
    {"id": "maml", "name": "MAML", "description": "二阶元学习更新全网络初始化；新任务仅用支持集标签适应。"},
    {"id": "calibrated", "name": "特征校准", "description": "分布对齐训练、基础类统计收缩及未标注查询集上的原型自适应。"},
]
LABELS = {
    "baseline": "Baseline", "maml": "MAML", "full": "完整特征校准",
    "calibrated": "完整特征校准", "alignment_only": "仅分布对齐训练",
    "mean": "仅均值校准", "variance": "仅方差校准", "prototype": "仅原型更新",
    "mean_only": "仅均值校准", "variance_only": "仅方差校准",
    "prototype_only": "仅原型更新", "mean_variance": "均值与方差校准",
    "no_alignment_full": "去除分布对齐训练", "graph": "白化＋图传播校准", "multiview": "双视图稳健校准", "wide": "较宽骨干＋度量训练", "wideview": "较宽骨干＋双视图校准", "continued": "同容量续训＋双视图",
    "transform_only": "仅基础类中心化", "no_transform_full": "去除特征中心化",
    "unbalanced_full": "去除均衡查询先验", "original_baseline": "原版 Baseline（32维）",
    "original_full": "原版完整校准（32维）",
}
CLASS_NAMES = {
    "nematode": "线虫", "king crab": "帝王蟹", "golden retriever": "金毛寻回犬",
    "malamute": "阿拉斯加雪橇犬", "dalmatian": "斑点狗", "African hunting dog": "非洲野犬",
    "lion": "狮子", "ant": "蚂蚁", "black-footed ferret": "黑足鼬", "bookshop": "书店",
    "crate": "木箱", "cuirass": "胸甲", "electric guitar": "电吉他", "hourglass": "沙漏",
    "mixing bowl": "搅拌碗", "school bus": "校车", "scoreboard": "记分牌", "theater curtain": "剧院幕布",
    "vase": "花瓶", "trifle": "水果奶油甜点",
}


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class LabFourService:
    def __init__(self, artifacts: Path):
        self.artifacts = Path(artifacts)
        # functional_call temporarily swaps module parameters. Inference must
        # not interleave on a shared model, even though FastAPI uses threads.
        self._slot = threading.BoundedSemaphore(1)
        self._images = None
        self._image_stamp = None
        self._thumbnails = {}

    def snapshot(self):
        empty = {
            "available": False, "status": "模型与实测结果尚未就绪", "title": "小样本学习与特征校准",
            "dataset": {"name": "mini-ImageNet", "image_size": 84, "train_classes": 64,
                        "val_classes": 16, "test_classes": 20, "training_images_per_class": 600,
                        "protocol_note": "训练、验证和测试类别互不重叠；查询标签只用于最终评分。"},
            "model": {"name": "ResNet12-Narrow", "channels": [8, 16, 24, 32], "parameters": 50352},
            "methods": METHODS, "comparisons": [], "ablations": [], "training": [],
            "limitations": [], "provenance": {"seed": 20261122, "checkpoint_sha256": ""},
        }
        result_path = self.artifacts / "results.json"
        if not result_path.is_file():
            return empty
        try:
            from experiments.lab04.artifact_audit import audit_publication
            audited = audit_publication(self.artifacts)
            result = audited["core"]
            protocol, model = result["protocol"], result["model"]
            if result.get("status") != "complete":
                return empty
            rows = []
            for row in result["rows"]:
                accuracy, ci95 = float(row["accuracy"]), float(row["ci95"])
                if not (math.isfinite(accuracy) and 0 <= accuracy <= 1 and math.isfinite(ci95) and ci95 >= 0):
                    raise ValueError("Invalid measured metric")
                rows.append({"method": row["method"], "label": LABELS.get(row["method"], row["method"]),
                             "shot": int(row["shots"]), "accuracy": accuracy, "ci95": ci95,
                             "episodes": int(row["episodes"]),
                             "latency_ms": float(row.get("mean_episode_seconds", 0)) * 1000,
                             "timing_scope": row.get("timing_scope", "unknown"),
                              "paired_delta": row.get("paired_delta"), "paired_ci95": row.get("paired_ci95"),
                              "transductive": bool(row.get("transductive", False)),
                              "balanced_queries": bool(row.get("balanced_queries", False))})
            weights = result["weights"]
            ready = audited["ready"]
            warm = result["training"]["warm_ce"]["epochs"]
            evaluation_count = protocol["evaluation_pool_per_class"]
            extras = result.get("study", {})
            history = []
            original_path = self.artifacts / "history" / "v1" / "results.json"
            if original_path.is_file():
                original = audit_publication(original_path.parent)["core"]
                history = [{"version": "v1", "label": "原始实验 · 32维骨干 / 100任务",
                    "model": original["model"], "protocol": original["protocol"],
                    "rows": [{"method": row["method"], "label": LABELS.get(row["method"], row["method"]),
                              "shot": row["shots"], "accuracy": row["accuracy"], "ci95": row["ci95"],
                              "episodes": row["episodes"], "timing_scope": row.get("timing_scope", "unknown")}
                             for row in original["rows"] if row["method"] in ("baseline", "maml", "full", "prototype_only")]}]
            calibration = result.get("calibration", {})
            graph_extension = None
            graph_path = self.artifacts / "graph_results.json"
            if graph_path.is_file():
                graph_extension = audited["graph"]
                if graph_extension.get("status") != "complete" or graph_extension.get("base_weights_sha256") != weights["aligned"]["sha256"]:
                    raise ValueError("Graph extension does not match the published backbone")
                for row in graph_extension["rows"]:
                    if not (math.isfinite(row["accuracy"]) and 0 <= row["accuracy"] <= 1):
                        raise ValueError("Invalid graph accuracy")
                    rows.append({"method": "graph", "label": LABELS["graph"], "shot":row["shots"],
                        "accuracy":row["accuracy"], "ci95":row["ci95"], "episodes":row["episodes"],
                        "paired_delta":row["paired_delta"], "paired_ci95":row["paired_ci95"],
                        "paired_vs_full":row["paired_vs_full"], "paired_vs_full_ci95":row["paired_vs_full_ci95"],
                        "transductive":True,"balanced_queries":True,"timing_scope":"not_recorded"})
            multiview_extension = None
            multiview_path = self.artifacts / "multiview_results.json"
            if multiview_path.is_file():
                multiview_extension = audited["multiview"]
                if multiview_extension.get("status") != "complete" or multiview_extension["base_weights_sha256"] != weights["aligned"]["sha256"]:
                    raise ValueError("Multi-view extension does not match backbone")
                for row in multiview_extension["rows"]:
                    if not (math.isfinite(row["accuracy"]) and 0 <= row["accuracy"] <= 1 and math.isfinite(row["ci95"]) and row["ci95"] >= 0):
                        raise ValueError("Invalid multi-view measurement")
                    rows.append({"method":"multiview","label":LABELS["multiview"],"shot":row["shots"],
                        "accuracy":row["accuracy"],"ci95":row["ci95"],"episodes":row["episodes"],
                        "paired_delta":row["paired_delta"],"paired_ci95":row["paired_ci95"],
                        "paired_vs_graph":row["paired_vs_graph"],"paired_vs_graph_ci95":row["paired_vs_graph_ci95"],
                        "paired_vs_full":row["paired_vs_full"],"paired_vs_full_ci95":row["paired_vs_full_ci95"],
                        "transductive":True,"balanced_queries":True,"decision_rule":row["decision_rule"],
                        "timing_scope":"inference_from_cached_features","latency_ms":row["mean_episode_seconds"]*1000})
            wide_extension = audited["wide"]
            if wide_extension is not None:
                for row in wide_extension["rows"]:
                    rows.append({"method":"wide","label":LABELS["wide"],"shot":row["shots"],
                        "accuracy":row["accuracy"],"ci95":row["ci95"],"episodes":row["episodes"],
                        "paired_delta":row["paired_delta"],"paired_ci95":row["paired_ci95"],
                        "paired_vs_graph":row["paired_vs_graph"],"paired_vs_graph_ci95":row["paired_vs_graph_ci95"],
                        "paired_vs_multiview":row["paired_vs_multiview"],"paired_vs_multiview_ci95":row["paired_vs_multiview_ci95"],
                        "raw_accuracy":row["raw_accuracy"],"transductive":True,"balanced_queries":True,
                        "timing_scope":row["timing_scope"],"latency_ms":row["mean_episode_seconds"]*1000})
            wideview_extension = audited.get("wideview")
            if wideview_extension is not None:
                for row in wideview_extension["rows"]:
                    rows.append({"method":"wideview","label":LABELS["wideview"],"shot":row["shots"],
                        **{k:row[k] for k in ("accuracy","ci95","episodes","paired_delta","paired_ci95",
                            "paired_vs_graph","paired_vs_graph_ci95","paired_vs_multiview","paired_vs_multiview_ci95",
                            "paired_vs_wide","paired_vs_wide_ci95")},
                        "transductive":True,"balanced_queries":True,"timing_scope":row["timing_scope"],
                        "latency_ms":row["mean_episode_seconds"]*1000})
            continued_extension = audited.get("continued")
            if continued_extension is not None:
                for row in continued_extension["rows"]:
                    rows.append({"method":"continued","label":LABELS["continued"],"shot":row["shots"],
                        **{k:row[k] for k in ("accuracy","ci95","episodes","paired_delta","paired_ci95",
                            "paired_vs_graph","paired_vs_graph_ci95","paired_vs_multiview","paired_vs_multiview_ci95",
                            "paired_vs_wide","paired_vs_wide_ci95","paired_vs_wideview","paired_vs_wideview_ci95")},
                        "transductive":True,"balanced_queries":True,"timing_scope":row["timing_scope"],"latency_ms":row["mean_episode_seconds"]*1000})
            edge_benchmark = audited["edge_benchmark"]
            return {
                **empty, "available": ready,
                "verification": {"verified": audited["verified"], "core_tasks": audited["episodes_recomputed"],
                                 "extension_tasks": audited["extension_episodes_recomputed"],
                                 "scope": "逐任务重算、配对样本与预测核对、选择锁及权重/先验/图片池校验"},
                "status": "模型与实测结果已就绪" if ready else "实测结果可读，模型文件尚未齐全",
                "dataset": {"name": protocol["dataset"], "image_size": protocol["image_size"][0],
                            "train_classes": protocol["training_classes"], "val_classes": protocol["validation_classes"],
                            "test_classes": protocol["test_classes"], "training_images_per_class": protocol["train_per_class"],
                            "protocol_note": f"64/16/20类别隔离；训练每类{protocol['train_per_class']}张；验证和测试池每类{evaluation_count}张。每任务5类、每类15张查询图片，1-shot和5-shot各{protocol['test_episodes_per_shot']}个固定任务。"},
                "model": {"name": model["name"], "channels": model["widths"],
                          "parameters": model["parameters_baseline"], "alignment_weight": result["training"]["aligned"]["alignment_weight"]},
                "comparisons": [r for r in rows if r["method"] in ("baseline", "maml", "full", "calibrated", "graph", "multiview", "wide", "wideview", "continued")],
                "paired_history": [r for r in rows if r["method"].startswith("original_")],
                "history": history, "study": extras, "calibration": calibration, "edge_benchmark": edge_benchmark,
                "diagnostics": audited["diagnostics"],
                "module_contrasts": audited["module_contrasts"],
                "continued_extension": {"available":continued_extension is not None,
                    "model":continued_extension["model"] if continued_extension else None,
                    "training":continued_extension["training"] if continued_extension else None,
                    "description":"与较宽模型同容量，从原权重额外16轮训练，验证选择第12轮；固定双视图推理设置，不是同训练预算比较。"},
                "wideview_extension": {"available":wideview_extension is not None,
                    "description":"同一较宽权重的原图/翻转平均与训练类内白化，不再训练，每图编码两次。"},
                "wide_extension": {"available":wide_extension is not None,
                    "model":wide_extension["model"] if wide_extension else None,
                    "training":wide_extension["training"] if wide_extension else None,
                    "description":"独立的较宽骨干和额外度量训练研究，不是同骨干校准消融。"},
                "multiview_extension": {"available":multiview_extension is not None,
                    "description":"对同一图片的原图与水平翻转提取特征；预处理和容量假设仅由验证集选择。"},
                "graph_extension": {"available":graph_extension is not None,
                    "description":"同一轻量骨干，白化与图传播的独立后续实验；与原方案使用同一批任务。"},
                "ablations": [r for r in rows if r["method"] not in ("baseline", "maml") and not r["method"].startswith("original_")],
                "training": [{"epoch": r["epoch"], "loss": r["ce_loss"], "accuracy": r["training_accuracy"],
                              "validation_1shot": r.get("validation_1shot_accuracy"),
                              "validation_5shot": r.get("validation_5shot_accuracy")} for r in warm],
                "limitations": [
                    f"使用{model['parameters_baseline']:,}参数的轻量ResNet12和单次训练种子；不能与标准宽度模型的论文分数直接比较。",
                    *(extras.get("limitations", [])),
                    "95%置信区间描述同一模型下的任务抽样变化，没有覆盖重新训练模型带来的差异。",
                    "原型更新联合使用未标注查询图片，属于转导设置；查询标签不参与推理。",
                    "原始结果文件的缓存特征耗时与MAML编码适应耗时不可直接排名；页面另列统一编码范围的CPU实测，现场推理也包含图像编码。",
                    "类别分数用于模型内相对比较；特征校准不等于预测概率已经可靠校准。",
                    "本模块验证图像小样本方法，与原股权实验共享网站入口；结果不代表企业风险识别效果。",
                ],
                "provenance": {"seed": protocol["seed"], "checkpoint_sha256": weights["aligned"]["sha256"],
                               "run_id": result["completed_utc"], "artifact_revision": audited["artifact_revision"]},
            }
        except (KeyError, ValueError, TypeError, OSError, StopIteration) as exc:
            raise WebError("lab04_results_invalid", "实验四结果文件无法核验，请重新生成实验结果。", 503) from exc

    def _thumbnail(self, index):
        import numpy as np
        from PIL import Image
        pool = self.artifacts / "demo_pool.npz"
        stamp = (pool.stat().st_size, pool.stat().st_mtime_ns)
        if self._image_stamp != stamp:
            with np.load(pool, allow_pickle=False) as data:
                self._images = data["images"].copy()
            self._image_stamp = stamp
            self._thumbnails.clear()
        index = int(index)
        if index not in self._thumbnails:
            buffer = io.BytesIO()
            Image.fromarray(self._images[index]).save(buffer, format="JPEG", quality=88)
            self._thumbnails[index] = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        return self._thumbnails[index]

    def run(self, payload):
        if not self._slot.acquire(blocking=False):
            raise WebError("busy", "实验四模型正在推理，请稍后重试。", 429)
        try:
            if not self.snapshot()["available"]:
                raise WebError("lab04_not_ready", "请先生成实验四模型与结果文件。", 503)
            try:
                from experiments.lab04.inference import run_episode
                raw = run_episode(self.artifacts, **payload)
                return self._adapt_episode(raw)
            except ImportError as exc:
                raise WebError("lab04_dependencies_missing", "实验四现场推理需要PyTorch和Pillow，请安装实验四依赖。", 503) from exc
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                raise WebError("lab04_inference_failed", "实验四推理失败，请核对模型文件和实验配置。", 503) from exc
        finally:
            self._slot.release()

    def _adapt_episode(self, raw):
        import numpy as np
        classes = [{"id": str(c["wnid"]), "name": CLASS_NAMES.get(c["name"].split(",")[0], c["name"])} for c in raw["classes"]]
        labels = {str(c["label"]): str(c["wnid"]) for c in raw["classes"]}
        points = raw["support"] + raw["query"]
        before = np.asarray([p.get("feature_before", p["feature"]) for p in points], dtype=float)
        after = np.asarray([p.get("feature_after", p["feature"]) for p in points], dtype=float)
        prototypes_before = np.asarray(raw["prototypes_before"], dtype=float)
        prototypes_after = np.asarray(raw["prototypes_after"], dtype=float)
        from .lab_four_projection import project
        projection = project(before, after, prototypes_before, prototypes_after)
        a, b = projection["before"], projection["after"]
        pa, pb = projection["prototypes_before"], projection["prototypes_after"]
        explained = projection["explained_variance"]
        plot_points = [{"id": p["id"], "class_id": labels[str(p["label"])],
                        "kind": "support" if i < len(raw["support"]) else "query",
                        "before": a[i].tolist(), "after": b[i].tolist()} for i, p in enumerate(points)]
        query = [{"id": q["id"], "image": self._thumbnail(q["image_index"]),
                  "predicted_id": labels[str(q["prediction"])], "target_id": labels[str(q["label"])],
                  "confidence": q["confidence"], "probabilities": q["probabilities"]} for q in raw["query"]]
        steps = [
            {"name": "任务采样", "detail": f"固定种子{raw['seed']}；5个未见类，每类{raw['shots']}张支持图片与15张互斥查询图片。"},
            {"name": "图像编码", "detail": f"84×84彩色图片通过轻量ResNet12，获得{len(raw['support'][0]['feature'])}维特征。"},
        ]
        if raw["method"] == "maml":
            steps.append({"name": "任务适应", "detail": "从二阶元训练的初始化出发，仅用支持集标签更新全网络，再预测查询图片。"})
        elif raw["method"] == "calibrated":
            if raw.get("config", {}).get("feature_center_strength", 0):
                steps.append({"name": "训练先验预处理", "detail": "应用训练类计算的固定特征中心；支持与查询使用同一变换，不利用查询标签或当前查询均值。"})
            if raw.get("config", {}).get("balanced_queries") and raw.get("transductive"):
                steps.append({"name": "均衡查询先验", "detail": "本任务每类查询数量相等，软分配使用均衡类别质量约束；这是假设，不是读取查询标签，真实不均衡任务不能直接套用。"})
            if raw.get("calibration") in ("graph", "multiview", "wide", "wideview", "continued"):
                if raw.get("config",{}).get("graph_weight",0) > 0:
                    steps.append({"name": "图传播校准", "detail": "训练类统计白化后，在当前支持与无标签查询上构建近邻图；仅支持标签作为传播种子，与原型分数融合。二维原型不是完整决策规则。"})
                else:
                    steps.append({"name":"均衡原型更新", "detail":"验证配置关闭图传播，仅用无标签查询的均衡软分配迭代更新支持类原型；关闭的模块不记作有效贡献。"})
            if raw.get("calibration") in ("multiview", "wideview", "continued"):
                steps.append({"name":"多视图表征", "detail":"原图和同图水平翻转使用同一冻结骨干编码；未引入额外标注图像，编码开销相应增加。"})
                if raw.get("config", {}).get("residual_covariance_mix", 0):
                    steps.append({"name":"训练类内白化", "detail":"基础训练类标签用于估计类内协方差，压低训练类内的无关变化；不使用当前query标签估计任何变换。"})
            if raw.get("calibration") in ("wide", "wideview", "continued"):
                steps.append({"name":"较宽自训骨干", "detail":"从原自训骨干扩宽后在基础训练类上额外做交叉熵与度量训练；容量和训练预算不同，提升不能全部归于校准模块。"})
            if raw.get("calibration") == "continued":
                steps.append({"name":"同容量追加训练", "detail":"保持较宽模型通道和参数量，额外16轮训练中按验证选择第12轮；没有根据测试结果重选分类配置。额外预算与旧较宽双视图分开记录。"})
            if raw.get("calibration") == "wideview":
                steps.append({"name":"较宽双视图增量", "detail":"使用原较宽模型的同一权重，不新增训练；原图与水平翻转分别归一化后平均，训练类内协方差只来自基础类。每图编码两次。"})
            if raw.get("decision_rule") == "balanced_capacity":
                steps.append({"name":"容量约束决策", "detail":"根据任务每类15张查询的已知设定，对整批分配类别；可能选取非最高分项，类别分数保留原值不改成确定概率。"})
            if raw.get("calibration") not in ("graph", "multiview", "wide", "wideview", "continued"):
                steps.append({"name": "特征统计校准", "detail": "按所选消融组合应用基础类均值/方差先验及原型更新；查询真实标签不会传给分类器。"})
        else:
            steps.append({"name": "最近原型分类", "detail": "按支持集标签求类均值，用到各原型的距离预测查询图片。"})
        steps.append({"name": "独立评分", "detail": "预测完成后才将预测类别与查询标签对照；单次任务不能替代多任务统计。"})
        return {
            "method": raw["method"], "shots": raw["shots"], "seed": raw["seed"], "way": raw["ways"],
            "accuracy": raw["accuracy"], "correct": sum(bool(q["correct"]) for q in raw["query"]),
            "total": len(query), "latency_ms": raw["inference_ms"], "classes": classes,
            "support": [{"id": p["id"], "class_id": labels[str(p["label"])], "image": self._thumbnail(p["image_index"])} for p in raw["support"]],
            "queries": query, "points": plot_points,
            "prototypes": [{"class_id": c["id"], "before": pa[i].tolist(), "after": pb[i].tolist()} for i, c in enumerate(classes)],
            "steps": steps, "scope": "mini-ImageNet未见类；本机CPU现场推理；教学研究模型",
            "checkpoint_sha256": raw["checkpoint_sha256"], "compute": "live",
            "pool_hash": raw.get("pool_hash"), "config": raw.get("config", {}),
            "study_id": raw.get("study_id", "v1"), "artifact_revision": raw.get("artifact_revision"),
            "balanced_queries": bool(raw.get("config", {}).get("balanced_queries", False) and raw.get("transductive")),
            "transductive": raw["transductive"], "metric": raw["metric"], "decision_rule":raw.get("decision_rule","score_argmax"),
            "feature_motion": raw["feature_motion"], "calibration": raw["calibration"],
            "projection": {"method":"PCA", "explained_variance":explained,
                           "feature_dimensions":int(before.shape[1]), "computed_after_prediction":True, "isotropic":True},
            "prototype_role": raw.get("prototype_role", "descriptive_class_centers" if raw["method"] == "maml" else "decision_prototypes"),
        }
