"""无第三方依赖的分类与评分指标；缺失类别的 AUC 返回 None。"""
import math


def evaluate(labels: list[int], scores: list[float], threshold: float = .5) -> dict:
    if not labels or len(labels) != len(scores):
        raise ValueError("标签与评分须非空且等长")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("标签须为 0/1")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in scores):
        raise ValueError("评分须为 [0,1] 内有限数")
    if not 0 <= threshold <= 1:
        raise ValueError("分类阈值须在 [0,1]")
    predictions = [int(p >= threshold) for p in scores]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f0 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    n = len(labels)
    return {"n": n, "positives": sum(labels), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp + tn) / n, "precision": precision, "recall": recall,
            "f1": f1, "macro_f1": (f1 + f0) / 2,
            "brier": sum((p - y) ** 2 for y, p in zip(labels, scores)) / n,
            "log_loss": -sum(y * math.log(max(1e-15, p)) + (1 - y) * math.log(max(1e-15, 1 - p))
                              for y, p in zip(labels, scores)) / n,
            "auc": roc_auc(labels, scores)}


def roc_auc(labels, scores):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    pairs = sorted(zip(scores, labels))
    rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += ((i + 1 + j) / 2) * sum(y for _, y in pairs[i:j])
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def reliability_bins(labels, scores, n_bins=10):
    out = []
    for i in range(n_bins):
        pairs = [(y, p) for y, p in zip(labels, scores) if min(int(p * n_bins), n_bins - 1) == i]
        if pairs:
            out.append({"bin": i, "count": len(pairs),
                        "mean_score": sum(p for _, p in pairs) / len(pairs),
                        "positive_fraction": sum(y for y, _ in pairs) / len(pairs)})
    return out
