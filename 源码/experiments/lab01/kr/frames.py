"""框架表示：框架 = 若干槽(slot)，每个槽含若干侧面(facet)，下层框架通过 is_a 继承上层框架。

侧面约定：值 / 默认 / 范围 / 单位
"""


class Frame:
    def __init__(self, name, is_a=None, slots=None):
        self.name = name
        self.is_a = is_a                # 上层框架（Frame 对象）
        self.slots = slots or {}        # 槽名 -> {侧面名: 侧面值}

    def get(self, slot, facet="值"):
        """沿 is_a 链查找槽值；取"值"时若没有填写，则用"默认"侧面（缺省推理）。"""
        frame = self
        while frame is not None:
            s = frame.slots.get(slot, {})
            if facet in s:
                return s[facet]
            if facet == "值" and "默认" in s:
                return s["默认"]
            frame = frame.is_a
        return None

    def set(self, slot, value, facet="值"):
        allowed = self.get(slot, "范围")
        if facet == "值" and allowed is not None and value not in allowed:
            raise ValueError(f"{self.name}.{slot} 的取值 {value} 不在范围 {allowed} 内")
        self.slots.setdefault(slot, {})[facet] = value

    def all_slots(self):
        names, frame = [], self
        while frame is not None:
            names += [s for s in frame.slots if s not in names]
            frame = frame.is_a
        return names

    def show(self):
        lines = [f"框架名：<{self.name}>" + (f"   (is_a <{self.is_a.name}>)" if self.is_a else "")]
        for slot in self.all_slots():
            value, unit = self.get(slot), self.get(slot, "单位")
            inherited = "" if "值" in self.slots.get(slot, {}) else "  [继承/缺省]"
            lines.append(f"    {slot}：{value}{unit or ''}{inherited}")
        return "\n".join(lines)


def build_company_frames(dataset):
    """构建 企业 → 上市公司 / 担保公司 的框架网络，并为每家企业生成实例框架。"""
    enterprise = Frame("企业", slots={
        "名称": {},
        "注册资本": {"单位": "万元"},
        "实际控制人": {"默认": "未识别"},
        "风险等级": {"默认": "低", "范围": ["低", "中", "高"]},
        "风险提示": {"默认": "无"},
        "关联交易披露": {"默认": "不强制"},
    })
    classes = {
        "企业": enterprise,
        "上市公司": Frame("上市公司", is_a=enterprise, slots={
            "关联交易披露": {"默认": "必须公告披露"},
            "监管机构": {"默认": "证监会/交易所"},
        }),
        "担保公司": Frame("担保公司", is_a=enterprise, slots={
            "风险等级": {"默认": "中"},
            "监管机构": {"默认": "地方金融监管局"},
        }),
    }
    instances = {}
    for cid, name, category, capital in dataset["companies"]:
        f = Frame(f"{name}", is_a=classes[category])
        f.set("名称", name)
        f.set("注册资本", capital)
        instances[cid] = f
    return classes, instances
