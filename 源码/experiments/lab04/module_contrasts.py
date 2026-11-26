"""Conditional module comparisons from already-audited, frozen paired tasks.

No inference, tuning or file mutation. Pointwise intervals are descriptive and
not a simultaneous family-wise claim about a set of post-hoc comparisons.
"""
from __future__ import annotations
import math
import statistics

# These comparisons keep their real information/budget differences visible.
CONTRASTS=(
    ('alignment','分布对齐训练','full','no_alignment_full',
     '同初始化、增强和CE预算的训练对照；都使用完整任务校准。不是同一检查点的推理开关。'),
    ('mean','基础类均值收缩','mean_only','transform_only',
     '在同一训练对齐骨干和固定中心化上加入均值收缩；不使用查询联合更新。'),
    ('variance','再加入对角方差','mean_variance','mean_only',
     '保持均值收缩与骨干，在这一条件下加入类条件方差尺度；不是所有配置的平均作用。'),
    ('prototype_balanced','原型更新＋均衡转导','full','mean_variance',
     '同时加入无标签查询原型更新与均衡分配，信息条件由归纳变为转导；不能全部归因于原型单一模块。'),
    ('statistics_on_prototype','在原型流程上加统计校准','full','prototype_only',
     '原型更新及均衡分配不变，联合加入均值和方差校准；这是一组组合改动，不拆分二者贡献。'),
    ('balance','均衡查询先验','full','unbalanced_full',
     '固定完整流程，将普通softmax替换为均衡软分配；只适用于已声明的类别均衡query协议。'),
)


def summarize_contrasts(reference):
    rows=[]
    for shot in (1,5):
        tasks=[value for (s,_),value in sorted(reference.items()) if s==shot]
        if not tasks:continue
        for identifier,label,method,baseline,scope in CONTRASTS:
            present=[method in r['scores'] and baseline in r['scores'] for r in tasks]
            if not any(present):continue
            if not all(present):raise ValueError('Unpaired module comparison; missing method in some tasks')
            values=[r['scores'][method] for r in tasks];control=[r['scores'][baseline] for r in tasks]
            if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in values+control):
                raise ValueError('Invalid frozen score in module comparison')
            differences=[a-b for a,b in zip(values,control)]
            mean=statistics.mean(differences)
            half=1.96*statistics.stdev(differences)/len(differences)**.5 if len(differences)>1 else 0.
            rows.append({'id':identifier,'label':label,'method':method,'reference':baseline,'scope':scope,'shot':shot,
                'episodes':len(tasks),'method_accuracy':statistics.mean(values),'reference_accuracy':statistics.mean(control),
                'paired_delta':mean,'paired_ci95':half,'ci95_low':mean-half,'ci95_high':mean+half,
                'wins':sum(v>0 for v in differences),'ties':sum(v==0 for v in differences),'losses':sum(v<0 for v in differences)})
    return {'rows':rows,
            'scope':'同一冻结模型实验的事后配对诊断；不改变预测，不用于反向选型。每行只解释指定对照条件。',
            'interval_note':'95%区间由逐任务差值重算，不是两个独立区间相减；未作多重比较校正，不证明跨训练种子或所有配置的因果效应。'}
