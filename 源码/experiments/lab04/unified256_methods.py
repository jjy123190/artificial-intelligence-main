"""Label-free episode predictors for the unified 256-D comparison (no query-label arguments)."""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import torch
from torch.nn import functional as F

from .models import maml_adapt
from .unified256_train import INNER_LR, INNER_STEPS

# Full model uses ONE image view and NO regional descriptor. Three optional
# extensions form a separate 2x2 factorial with that same full model as control.
@dataclass(frozen=True)
class Method:
    encoder: str = "da"
    mean: bool = True
    variance: bool = True
    update: bool = True
    balance: bool = True
    flip: bool = False
    local: bool = False


METHODS = {
    "baseline": Method("ce", False, False, False, False),
    "full": Method(),
    "no_alignment": Method("ce"),
    "no_mean": Method(mean=False),
    "no_variance": Method(variance=False),
    "no_update": Method(update=False),
    "no_balance": Method(balance=False),
    "full_flip": Method(flip=True),
    "full_local": Method(local=True),
    "full_flip_local": Method(flip=True, local=True),
}
MAML_NAMES = ("maml", "maml_no_adapt")

# Fixed grid on validation, one shared choice for both shots and every ablation.
# Tie break is the listed order. No rule is tuned on the test tasks.
RULES = (
    {"mean_weight": .10, "variance_blend": .20, "temperature": .12},
    {"mean_weight": .15, "variance_blend": .35, "temperature": .085},
    {"mean_weight": .20, "variance_blend": .50, "temperature": .065},
)
NEIGHBORS, PRIOR_DOF, UPDATES, SINKHORN, QUERY_WEIGHT = 3, 4., 8, 40, .50
LOCAL_WEIGHT = .15


def protocol_config():
    return {"methods": {k: asdict(v) for k, v in METHODS.items()},
            "rules": list(RULES), "neighbors": NEIGHBORS, "prior_dof": PRIOR_DOF,
            "prototype_steps": UPDATES, "query_weight": QUERY_WEIGHT, "sinkhorn_steps": SINKHORN,
            "local": {"stage": 3, "grid": [3, 3], "weight": LOCAL_WEIGHT, "scale_floor": .03},
            "maml": {"inner_steps": INNER_STEPS, "inner_lr": INNER_LR, "normalization": "frozen BN stats"},
            "pixels": "84x84 original RGB, /255 then mean [.485,.456,.406], std [.229,.224,.225]",
            "precision": "float32, channels_last; TF32 permitted",
            "centering": "L2-normalize raw GAP, subtract training feature mean, L2-normalize again"}


def task_hash(row, support_labels):
    fields = [row["shots"], row["episode"], row["support_ids"], list(map(int, support_labels)), row["query_ids"]]
    return hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass
class EpisodeBatch:
    support: np.ndarray
    labels: np.ndarray
    query: np.ndarray
    hashes: list


def index_episodes(split, rows, shots):
    """Build model inputs using IDs and SUPPORT labels only. Do not unpack query labels."""
    lookup = {v: i for i, v in enumerate(split.ids)}
    rows = sorted((r for r in rows if r["shots"] == shots), key=lambda r: r["episode"])
    if [r["episode"] for r in rows] != list(range(600)):
        raise ValueError("Expected all 600 released episodes")
    supports, labels, queries, hashes = [], [], [], []
    for row in rows:
        si = [lookup[k] for k in row["support_ids"]]
        qi = [lookup[k] for k in row["query_ids"]]
        raw = split.labels[si].tolist()
        order = list(dict.fromkeys(raw))
        y = [order.index(v) for v in raw]
        if (len(si) != 5 * shots or len(qi) != 75 or len(set(si + qi)) != len(si) + 75
                or np.bincount(y).tolist() != [shots] * 5):
            raise ValueError("Invalid support/query IDs or support counts")
        supports.append(si)
        queries.append(qi)
        labels.append(y)
        hashes.append(task_hash(row, y))
    return EpisodeBatch(*(np.asarray(x, dtype=np.int64) for x in (supports, labels, queries)), hashes)


def normalized_pixels(images, device):
    x = torch.as_tensor(images, device=device).permute(0, 3, 1, 2).float().div_(255)
    mean = x.new_tensor([.485, .456, .406])[None, :, None, None]
    std = x.new_tensor([.229, .224, .225])[None, :, None, None]
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


@torch.inference_mode()
def descriptors(model, images, device, *, flip=False, local=False):
    model.eval()
    out = {}
    def record(key, x):
        out.setdefault(key, []).append(x.contiguous().cpu())
    def forward(x):
        for block in list(model.encoder)[:3]:
            x = block(x)
        return model.encoder[3](x).mean((-2, -1)), F.adaptive_avg_pool2d(x, 3) if local else None
    for start in range(0, len(images), 64):
        x = normalized_pixels(images[start:start + 64], device)
        a, pa = forward(x)
        record("single", a)
        if local:
            record("patch", pa.flatten(2).transpose(1, 2))
        if flip:
            b, pb = forward(x.flip(-1))
            record("flip", (a + b) * .5)
            if local:
                record("flip_patch", ((pa + pb.flip(-1)) * .5).flatten(2).transpose(1, 2))
    return {key: torch.cat(values) for key, values in out.items()}


def fit_prior(raw, labels):
    x = F.normalize(raw.float(), dim=-1)
    center = x.mean(0)
    x = F.normalize(x - center, dim=-1)
    y = torch.as_tensor(labels, device=x.device, dtype=torch.long)
    if torch.bincount(y).tolist() != [600] * 64:
        raise ValueError("Fit priors only on the verified 64-class base split")
    groups = [x[y == i] for i in range(64)]
    return {"center": center, "means": torch.stack([g.mean(0) for g in groups]),
            "variances": torch.stack([g.var(0, unbiased=False) for g in groups])}


def transform(raw, prior):
    return F.normalize(F.normalize(raw.float(), dim=-1) - prior["center"], dim=-1)


def probabilities(logits, balanced):
    if not balanced:
        return logits.softmax(-1)
    # Normalize each task separately. Only the design prior (equal class mass)
    # is used; no true labels, no target histogram read from data.
    weights = (logits - logits.amax(-1, keepdim=True)).clamp_min(-70.).exp()
    for _ in range(SINKHORN):
        weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-30)
        weights = weights / weights.sum(2, keepdim=True).clamp_min(1e-30)
    return weights


def local_bonus(support, labels, query):
    """Symmetric best-patch support/query cosine; no query-to-query labels/links."""
    b, s, p, d = support.shape
    q = query.shape[1]
    support = F.normalize(support - support.mean(2, keepdim=True), dim=-1)
    query = F.normalize(query - query.mean(2, keepdim=True), dim=-1)
    pair = (query.reshape(b, q*p, d) @ support.reshape(b, s*p, d).transpose(1, 2)).reshape(b, q, p, s, p)
    sim = (pair.amax(-1).mean(2) + pair.amax(2).mean(-1)) * .5
    hot = F.one_hot(labels, 5).to(query.dtype)
    scores = sim @ (hot / hot.sum(1, keepdim=True))
    return LOCAL_WEIGHT * (scores - scores.mean(-1, keepdim=True)) / scores.std(-1, keepdim=True, unbiased=False).clamp_min(.03)


@torch.inference_mode()
def calibrated(support, labels, query, prior, rule, method, bonus=None):
    """Batched 5-way predictor; update and balanced final assignment are INDEPENDENT."""
    hot = F.one_hot(labels, 5).to(query.dtype)
    counts = hot.sum(1).unsqueeze(-1)
    raw = hot.transpose(1, 2) @ support / counts
    if method.encoder == "ce" and not any((method.mean, method.variance, method.update, method.balance)):
        return (query @ F.normalize(raw, dim=-1).transpose(1, 2)).argmax(-1)
    means, variances = prior["means"], prior["variances"]
    near = (raw.unsqueeze(2) - means).square().sum(-1).topk(NEIGHBORS, largest=False).indices
    near_means = means[near]
    prior_mean = near_means.mean(2)
    prior_var = (variances[near] + (near_means - prior_mean.unsqueeze(2)).square()).mean(2)
    shared = variances.mean().clamp_min(1e-6)
    var = shared
    if method.variance:
        assigned = raw.gather(1, labels.unsqueeze(-1).expand_as(support))
        ss = hot.transpose(1, 2) @ ((support - assigned).square())
        shrunk = (ss + PRIOR_DOF * prior_var) / (counts - 1 + PRIOR_DOF)
        floor = (.1 * variances.mean(0)).clamp_min(1e-6)
        v = rule["variance_blend"]
        var = (1 - v) * shared + v * torch.maximum(shrunk, floor)
    initial = F.normalize(raw.lerp(prior_mean, rule["mean_weight"]) if method.mean else raw, dim=-1)
    proto = initial
    for step in range((UPDATES if method.update else 0) + 1):
        dist2 = (query.unsqueeze(2) - proto.unsqueeze(1)).square()
        logits = -(dist2 / (var.unsqueeze(1) if var.ndim else var)).mean(-1)
        if bonus is not None:
            logits = logits + bonus
        weights = probabilities(logits / rule["temperature"], method.balance)
        if method.update and step < UPDATES:
            proto = F.normalize(counts * initial + QUERY_WEIGHT * (weights.transpose(1, 2) @ query), dim=-1)
    return weights.argmax(-1)


@torch.inference_mode()
def metric_predictions(raw, priors, tasks, rule, methods, device):
    results = {name: [] for name in methods}
    cache = {}
    for name, spec in methods.items():
        view = "flip" if spec.flip else "single"
        key = (spec.encoder, view)
        if key not in cache:
            prior = {k: v.to(device) for k, v in priors[key].items()}
            cache[key] = transform(raw[spec.encoder][view].to(device), prior), prior
    for i in range(0, len(tasks.labels), 40):
        sl = slice(i, i + 40)
        si, qi = [torch.as_tensor(a[sl], device=device, dtype=torch.long) for a in (tasks.support, tasks.query)]
        y = torch.as_tensor(tasks.labels[sl], device=device, dtype=torch.long)
        for name, spec in methods.items():
            x, prior = cache[(spec.encoder, "flip" if spec.flip else "single")]
            patch_logits = None
            if spec.local:
                patches = raw[spec.encoder]["flip_patch" if spec.flip else "patch"]
                patch_logits = local_bonus(patches[tasks.support[sl]].to(device), y, patches[tasks.query[sl]].to(device))
            pred = calibrated(x[si], y, x[qi], prior, rule, spec, patch_logits)
            results[name].append(pred.cpu().numpy())
    return {k: np.concatenate(v) for k, v in results.items()}


def maml_predictions(model, images, tasks, device, limit=None):
    if model.training or any(m.training for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)):
        raise ValueError("MAML must use frozen running statistics for each independent episode")
    pred, zero, evidence = [], [], []
    for i in range(len(tasks.labels) if limit is None else limit):
        sx = normalized_pixels(images[tasks.support[i]], device)
        qx = normalized_pixels(images[tasks.query[i]], device)
        y = torch.as_tensor(tasks.labels[i], device=device, dtype=torch.long)
        with torch.enable_grad():
            fast = maml_adapt(model, sx, y, learning_rate=INNER_LR, steps=INNER_STEPS, second_order=False)
        with torch.no_grad():
            before = model(qx)
            after = torch.func.functional_call(model, fast, (qx,))
            zero.append(before.argmax(-1).cpu().numpy())
            pred.append(after.argmax(-1).cpu().numpy())
            first = "encoder.0.convolutions.0.weight"
            delta = torch.stack(((fast[first] - model.encoder[0].convolutions[0].weight).norm(),
                                 (fast["classifier.weight"] - model.classifier.weight).norm())).tolist()
            evidence.append({"first_conv_delta_l2": delta[0], "head_delta_l2": delta[1]})
        del fast
    return np.stack(pred), np.stack(zero), evidence

