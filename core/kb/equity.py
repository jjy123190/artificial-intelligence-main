"""有环交叉持股的经济权益：解 u_j = d_j + sum_i H_ij u_i。

只计算目标的上游企业子图，避免无关封闭环使正常目标无解。部分主元高斯消元，
不使用伪逆强造唯一解；未知直接股东份额单列，经济权益不等于表决权或控制权。
"""
from __future__ import annotations
import math
import time
from .domain import EquityCase


def _solve(matrix, rhs, tolerance=1e-11):
    n = len(matrix)
    a = [list(row) + list(values) for row, values in zip(matrix, rhs)]
    columns = len(rhs[0]) if rhs else 0
    smallest = 1.0
    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(a[r][i]))
        magnitude = abs(a[pivot][i])
        if magnitude < tolerance:
            raise ValueError("封闭或近乎100%交叉持股导致方程奇异，无法给出唯一权益")
        smallest = min(smallest, magnitude)
        a[i], a[pivot] = a[pivot], a[i]
        scale = a[i][i]
        a[i] = [v / scale for v in a[i]]
        for r in range(n):
            if r == i:
                continue
            factor = a[r][i]
            if factor:
                a[r] = [x - factor * y for x, y in zip(a[r], a[i])]
    return [row[n:n + columns] for row in a], smallest


def economic_equity(case: EquityCase, target: str) -> dict:
    start = time.perf_counter()
    case.validate()
    if target not in {e.eid for e in case.companies}:
        raise ValueError("经济权益目标必须为案例中的企业")
    upstream = {target}
    company_ids = {e.eid for e in case.companies}
    changed = True
    while changed:
        old = len(upstream)
        upstream.update(h.holder for h in case.holdings if h.target in upstream and h.holder in company_ids)
        changed = len(upstream) != old
    companies = sorted(upstream)
    people = sorted({h.holder for h in case.holdings if h.target in upstream and h.holder not in company_ids})
    indices = {c: i for i, c in enumerate(companies)}
    person_indices = {p: i for i, p in enumerate(people)}
    n, k = len(companies), len(people) + 1
    a = [[float(i == j) for j in range(n)] for i in range(n)]
    rhs = [[0.0] * k for _ in range(n)]
    incoming = {c: 0.0 for c in companies}
    for h in case.holdings:
        if h.target not in upstream:
            continue
        row = indices[h.target]
        incoming[h.target] += h.ratio
        if h.holder in company_ids:
            a[row][indices[h.holder]] -= h.ratio
        else:
            rhs[row][person_indices[h.holder]] += h.ratio
    for c, total in incoming.items():
        rhs[indices[c]][-1] = max(0.0, 1 - total)
    base = {"target_id": target, "companies_solved": n, "method": "partial_pivot_linear_system",
            "formula": "u_j = d_j + sum_i H_ij*u_i", "upstream_companies": companies,
            "note": "教学经济权益模型；非法律受益人认定，非表决权。未登记持股计入未知股东份额。"}
    try:
        solution, pivot = _solve(a, rhs)
    except ValueError as exc:
        return {**base, "status": "indeterminate", "reason": str(exc), "owners": [],
                "unknown_share": None, "residual": None, "elapsed_ms": (time.perf_counter() - start) * 1000}
    residual = max(abs(math.fsum(a[i][j] * solution[j][c] for j in range(n)) - rhs[i][c])
                   for i in range(n) for c in range(k))
    values = solution[indices[target]]
    if residual > 1e-8 or any(not math.isfinite(v) or v < -1e-8 or v > 1 + 1e-8 for v in values):
        return {**base, "status": "indeterminate", "reason": "方程数值不稳定或权益范围不合法",
                "owners": [], "unknown_share": None, "residual": residual,
                "elapsed_ms": (time.perf_counter() - start) * 1000}
    owners = [{"entity_id": p, "name": case.name(p), "share": min(1.0, max(0.0, values[i]))}
              for i, p in enumerate(people) if values[i] > 1e-12]
    return {**base, "status": "solved", "owners": owners, "unknown_share": max(0.0, values[-1]),
            "total_share": math.fsum(values), "residual": residual, "minimum_pivot": pivot,
            "numerical_warning": pivot < 1e-6, "elapsed_ms": (time.perf_counter() - start) * 1000}
