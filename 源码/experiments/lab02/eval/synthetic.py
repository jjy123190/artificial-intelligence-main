"""合成企业风险测度实验，非真实企业数据集。

先生成不可见的二元风险状态，再由该状态生成带重叠分布的四个指标；标签不是
待评估规则的输出。生成族接近朴素贝叶斯假设，因此不能用此实验宣称行业优势。
"""
from __future__ import annotations
import random
from dataclasses import dataclass
from core.kb.features import FEATURES, RiskObservation

GENERATOR = {
    "label_prior": .34,
    "indicator_p_high_given_label": {"opacity": (.20, .80), "related_trade": (.14, .77),
                                     "guarantee": (.16, .70), "cycle": (.09, .50)},
    "measurement_beta_low": (2.0, 8.0),
    "measurement_beta_high": (8.0, 2.0),
    "label_meaning": "合成的高复核需求潜在状态，不是真实违法、违约或监管标签",
}
SPLIT_OFFSETS = {"train": 101, "validation": 211, "test": 307}


@dataclass(frozen=True)
class Sample:
    observation: RiskObservation
    label: int


def generate(n: int, seed: int, split: str) -> list[Sample]:
    if n < 1 or split not in SPLIT_OFFSETS:
        raise ValueError("样本数必须为正数，split 为 train/validation/test")
    rng = random.Random(seed * 1000003 + SPLIT_OFFSETS[split])
    out = []
    for i in range(n):
        y = int(rng.random() < GENERATOR["label_prior"])
        values = {}
        for f in FEATURES:
            high = rng.random() < GENERATOR["indicator_p_high_given_label"][f][y]
            a, b = GENERATOR["measurement_beta_high" if high else "measurement_beta_low"]
            values[f] = rng.betavariate(a, b)
        sample_id = f"s{seed}-{split}-{i:06d}"
        out.append(Sample(RiskObservation(sample_id, values, source="合成企业测度，非真实样本"), y))
    return out


def perturb(samples: list[Sample], *, sigma: float = 0.0, missing: float = 0.0, seed: int = 0) -> list[Sample]:
    """固定基础样本、标签与随机噪声方向；改变强度得到配对压力实验。"""
    if sigma < 0 or not 0 <= missing <= 1:
        raise ValueError("sigma≥0，missing∈[0,1]")
    rng = random.Random(seed)
    out = []
    for sample in samples:
        values = {}
        for f in FEATURES:
            noise, drop = rng.gauss(0, 1), rng.random()
            original = sample.observation.values.get(f)
            values[f] = (None if original is None or drop < missing
                         else min(1.0, max(0.0, original + sigma * noise)))
        o = sample.observation
        out.append(Sample(RiskObservation(o.company_id, values,
                    source=f"合成测度，Gaussian σ={sigma}; 缺测率={missing}", quality=o.quality), sample.label))
    return out


def assert_disjoint(*splits: list[Sample]) -> None:
    seen = set()
    for split in splits:
        ids = {s.observation.company_id for s in split}
        if len(ids) != len(split) or seen & ids:
            raise ValueError("训练/验证/测试 ID 重复或交叉")
        seen |= ids
