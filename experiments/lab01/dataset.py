"""数据层：演示数据集（虚构企业）+ 大规模合成数据生成 + CSV 导出。

数据集统一为 dict，字段如下：
    persons    : [(id, 姓名)]
    companies  : [(id, 名称, 类型, 注册资本万元)]    类型 ∈ {企业, 上市公司, 担保公司}
    holdings   : [(股东id, 被投企业id, 持股比例)]
    guarantees : [(担保方id, 被担保方id, 金额万元)]
    trades     : [(卖方id, 买方id, 金额万元)]
    kinships   : [(自然人1, 自然人2, 关系)]
"""
import csv
import os
import random


def demo_dataset():
    """课堂演示用的小数据集，人为埋入 4 类风险场景：
    1. 多层控股 + 合并持股：王建国 通过 华鑫控股 / 华鑫实业 间接控制上市公司 恒通科技
    2. 一致行动人：王建国 与配偶 李秀英 合计持有 远航贸易 55%
    3. 隐性关联交易：远航贸易 向 恒通科技 大额供货，二者实控人相同
    4. 循环持股（鼎丰投资→鼎丰建设→瑞祥建材→鼎丰投资）与 担保圈（鼎丰建设→天润商贸→博远咨询→鼎丰建设）
    """
    persons = [
        ("P1", "王建国"), ("P2", "李秀英"), ("P3", "陈志强"), ("P4", "赵敏"),
    ]
    companies = [
        ("C1", "华鑫控股集团", "企业", 50000),
        ("C2", "华鑫实业有限公司", "企业", 20000),
        ("C3", "恒通科技股份有限公司", "上市公司", 80000),
        ("C4", "远航贸易有限公司", "企业", 5000),
        ("C5", "盛达物流有限公司", "企业", 3000),
        ("C6", "鼎丰投资有限公司", "企业", 30000),
        ("C7", "鼎丰建设有限公司", "企业", 15000),
        ("C8", "瑞祥建材有限公司", "企业", 8000),
        ("C9", "金桥融资担保有限公司", "担保公司", 10000),
        ("C10", "天润商贸有限公司", "企业", 6000),
        ("C11", "博远咨询有限公司", "企业", 1000),
        ("C12", "星辰电子股份有限公司", "上市公司", 60000),
    ]
    holdings = [
        ("P1", "C1", 0.80), ("C1", "C2", 0.65), ("C2", "C3", 0.35), ("C1", "C3", 0.20),
        ("P1", "C4", 0.30), ("P2", "C4", 0.25), ("C4", "C5", 0.90),
        ("P3", "C6", 0.70), ("C6", "C7", 0.60), ("C7", "C8", 0.40), ("C8", "C6", 0.30),
        ("P4", "C10", 0.51), ("P4", "C11", 0.60), ("P4", "C9", 0.30), ("C1", "C9", 0.40),
        ("C6", "C12", 0.30), ("P4", "C12", 0.10),
    ]
    guarantees = [
        ("C7", "C10", 3000), ("C10", "C11", 2000), ("C11", "C7", 2500), ("C9", "C12", 1000),
    ]
    trades = [
        ("C4", "C3", 5000), ("C5", "C2", 2000), ("C10", "C12", 800),
    ]
    kinships = [("P1", "P2", "配偶")]
    return dict(persons=persons, companies=companies, holdings=holdings,
                guarantees=guarantees, trades=trades, kinships=kinships)


def synthetic_dataset(n_layers=8, width=50, n_persons=30, max_holders=3, seed=42):
    """生成分层股权网络（用于效率实验）。第 0 层为自然人，其余每层 width 家企业，
    每家企业有 1~max_holders 个来自上一层的股东。"""
    rng = random.Random(seed)
    persons = [(f"P{i}", f"自然人{i}") for i in range(n_persons)]
    layers = [[p for p, _ in persons]]
    companies, holdings = [], []
    for layer in range(1, n_layers + 1):
        ids = [f"L{layer}C{i}" for i in range(width)]
        for cid in ids:
            companies.append((cid, f"企业{cid}", "企业", rng.randint(100, 10000)))
            holders = rng.sample(layers[-1], rng.randint(1, max_holders))
            remain = 1.0
            for h in holders:
                r = round(rng.uniform(0.2, 0.7) * remain, 3)
                holdings.append((h, cid, r))
                remain -= r
        layers.append(ids)
    return dict(persons=persons, companies=companies, holdings=holdings,
                guarantees=[], trades=[], kinships=[])


def save_csv(dataset, out_dir):
    headers = dict(
        persons=["id", "姓名"], companies=["id", "名称", "类型", "注册资本万元"],
        holdings=["股东", "被投企业", "持股比例"], guarantees=["担保方", "被担保方", "金额万元"],
        trades=["卖方", "买方", "金额万元"], kinships=["自然人1", "自然人2", "关系"],
    )
    os.makedirs(out_dir, exist_ok=True)
    for key, rows in dataset.items():
        with open(os.path.join(out_dir, f"{key}.csv"), "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(headers[key])
            w.writerows(rows)
