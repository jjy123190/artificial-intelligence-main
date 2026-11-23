"""Adapt verified experiment files to the course website without eager ML imports."""
from __future__ import annotations

import base64
import io
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
    "no_alignment_full": "去除分布对齐训练",
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
            result = _read(result_path)
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
                             "timing_scope": row.get("timing_scope", "unknown")})
            weights = result["weights"]
            ready = all((self.artifacts / weights[k]["path"]).is_file() for k in ("baseline", "aligned", "maml"))
            ready = ready and (self.artifacts / result["demo_pool"]["path"]).is_file()
            warm = result["training"]["warm_ce"]["epochs"]
            evaluation_count = protocol["evaluation_pool_per_class"]
            return {
                **empty, "available": ready,
                "status": "模型与实测结果已就绪" if ready else "实测结果可读，模型文件尚未齐全",
                "dataset": {"name": protocol["dataset"], "image_size": protocol["image_size"][0],
                            "train_classes": protocol["training_classes"], "val_classes": protocol["validation_classes"],
                            "test_classes": protocol["test_classes"], "training_images_per_class": protocol["train_per_class"],
                            "protocol_note": f"64/16/20类别隔离；训练每类{protocol['train_per_class']}张；验证和测试池每类{evaluation_count}张。每任务5类、每类15张查询图片，1-shot和5-shot各{protocol['test_episodes_per_shot']}个固定任务。"},
                "model": {"name": model["name"], "channels": model["widths"],
                          "parameters": model["parameters_baseline"], "alignment_weight": result["training"]["aligned"]["alignment_weight"]},
                "comparisons": [r for r in rows if r["method"] in ("baseline", "maml", "full", "calibrated")],
                "ablations": [r for r in rows if r["method"] not in ("baseline", "maml")],
                "training": [{"epoch": r["epoch"], "loss": r["ce_loss"], "accuracy": r["training_accuracy"]} for r in warm],
                "limitations": [
                    "使用窄通道、约5万参数的ResNet12和单次训练种子；不能与标准宽度模型的论文分数直接比较。",
                    "95%置信区间描述同一模型下的任务抽样变化，没有覆盖重新训练模型带来的差异。",
                    "原型更新联合使用未标注查询图片，属于转导设置；查询标签不参与推理。",
                    "评测表中原型方法计缓存特征推理时间，MAML计图像编码与适应时间，二者不可直接作速度排名。现场推理统一包含图像编码。",
                    "类别分数用于模型内相对比较；特征校准不等于预测概率已经可靠校准。",
                    "本模块验证图像小样本方法，与原股权实验共享网站入口；结果不代表企业风险识别效果。",
                ],
                "provenance": {"seed": protocol["seed"], "checkpoint_sha256": weights["aligned"]["sha256"],
                               "run_id": result["completed_utc"]},
            }
        except (KeyError, ValueError, TypeError, OSError) as exc:
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
        # A single PCA basis and extent make the two plots comparable. This is
        # visualization after prediction; its input never enters the classifier.
        together = np.concatenate((before, after, prototypes_before, prototypes_after))
        center = together.mean(axis=0)
        _, _, vectors = np.linalg.svd(together - center, full_matrices=False)
        basis = vectors[:2].T
        for col in range(basis.shape[1]):
            if basis[np.argmax(np.abs(basis[:, col])), col] < 0:
                basis[:, col] *= -1
        projected = (together - center) @ basis
        minimum, maximum = projected.min(axis=0), projected.max(axis=0)
        projected = 0.06 + 0.88 * (projected - minimum) / np.maximum(maximum - minimum, 1e-8)
        if not np.isfinite(projected).all():
            raise ValueError("Non-finite visualization")
        count = len(points)
        a, b = projected[:count], projected[count:count * 2]
        pa = projected[count * 2:count * 2 + len(classes)]
        pb = projected[count * 2 + len(classes):]
        plot_points = [{"id": p["id"], "class_id": labels[str(p["label"])],
                        "kind": "support" if i < len(raw["support"]) else "query",
                        "before": a[i].tolist(), "after": b[i].tolist()} for i, p in enumerate(points)]
        query = [{"id": q["id"], "image": self._thumbnail(q["image_index"]),
                  "predicted_id": labels[str(q["prediction"])], "target_id": labels[str(q["label"])],
                  "confidence": q["confidence"], "probabilities": q["probabilities"]} for q in raw["query"]]
        steps = [
            {"name": "任务采样", "detail": f"固定种子{raw['seed']}；5个未见类，每类{raw['shots']}张支持图片与15张互斥查询图片。"},
            {"name": "图像编码", "detail": "84×84彩色图片通过窄通道ResNet12，获得32维特征。"},
        ]
        if raw["method"] == "maml":
            steps.append({"name": "任务适应", "detail": "从二阶元训练的初始化出发，仅用支持集标签更新全网络，再预测查询图片。"})
        elif raw["method"] == "calibrated":
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
            "transductive": raw["transductive"], "metric": raw["metric"],
            "feature_motion": raw["feature_motion"], "calibration": raw["calibration"],
            "prototype_role": raw.get("prototype_role", "descriptive_class_centers" if raw["method"] == "maml" else "decision_prototypes"),
        }
