"""Failure-prediction metrics with grouped-bootstrap hooks."""
from __future__ import annotations


def binary_metrics(labels, scores, n_bins: int = 10) -> dict:
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    import numpy as np
    y, p = np.asarray(labels,dtype=int), np.clip(np.asarray(scores,dtype=float),0,1)
    result = {"n":int(len(y)),"auroc":None,"auprc":None,"brier":None,"ece":None}
    if len(set(y.tolist())) > 1:
        result["auroc"] = float(roc_auc_score(y,p))
        result["auprc"] = float(average_precision_score(y,p))
    if len(y):
        result["brier"] = float(brier_score_loss(y,p))
        bins=np.linspace(0,1,n_bins+1); ece=0.0
        for low,high in zip(bins[:-1],bins[1:]):
            selected=(p>=low)&((p<high) if high<1 else (p<=high))
            if selected.any(): ece+=float(selected.mean()*abs(y[selected].mean()-p[selected].mean()))
        result["ece"]=ece
    return result


def grouped_bootstrap_metric(labels, scores, groups, metric="auroc", repeats=1000, seed=17):
    """Bootstrap independent source samples, keeping all their semantic variants together."""
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    y,p,g=np.asarray(labels),np.asarray(scores),np.asarray(groups)
    unique=np.unique(g); rng=np.random.default_rng(seed); estimates=[]
    fn={"auroc":roc_auc_score,"auprc":average_precision_score}.get(metric)
    if fn is None: raise ValueError("metric must be auroc or auprc")
    for _ in range(repeats):
        sampled=rng.choice(unique,size=len(unique),replace=True)
        idx=np.concatenate([np.where(g==group)[0] for group in sampled])
        if len(np.unique(y[idx]))<2: continue
        estimates.append(fn(y[idx],p[idx]))
    if not estimates:
        return {"mean": None, "ci95": None, "valid_replicates": 0, "requested_replicates": repeats}
    return {"mean":float(np.mean(estimates)),"ci95":[float(np.quantile(estimates,.025)),float(np.quantile(estimates,.975))],"valid_replicates":len(estimates),"requested_replicates":repeats}


def grouped_bootstrap_binary_metrics(labels, scores, groups, repeats=1000, seed=17, n_bins=10):
    """Binary metrics and grouped percentile CIs; one resampling unit per image."""
    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    y, p, g = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float), np.asarray(groups)
    if not (len(y) == len(p) == len(g)):
        raise ValueError("labels, scores, and groups must have equal lengths")
    base = binary_metrics(y, p, n_bins=n_bins)
    base["prevalence"] = float(y.mean()) if len(y) else None
    base["bootstrap"] = {}
    unique = np.unique(g)
    rng = np.random.default_rng(seed)
    values = {name: [] for name in ("auroc", "auprc", "brier")}
    if len(unique):
        group_indices = {group: np.flatnonzero(g == group) for group in unique}
        for _ in range(repeats):
            sampled = rng.choice(unique, size=len(unique), replace=True)
            idx = np.concatenate([group_indices[group] for group in sampled])
            yy, pp = y[idx], p[idx]
            if len(np.unique(yy)) > 1:
                values["auroc"].append(float(roc_auc_score(yy, pp)))
                values["auprc"].append(float(average_precision_score(yy, pp)))
            if len(yy):
                values["brier"].append(float(brier_score_loss(yy, pp)))
    for name, estimates in values.items():
        base["bootstrap"][name] = {
            "ci95": [float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))] if estimates else None,
            "valid_replicates": len(estimates), "requested_replicates": repeats,
        }
    base["bootstrap"]["resamples_with_both_classes"] = len(values["auroc"])
    base["bootstrap"]["unique_image_groups"] = int(len(unique))
    return base


def grouped_bootstrap_auc_difference(labels, scores_a, scores_b, groups, repeats=1000, seed=17):
    """Paired image-group bootstrap for AUROC(A)-AUROC(B)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    y, a, b, g = (np.asarray(x) for x in (labels, scores_a, scores_b, groups))
    if not (len(y) == len(a) == len(b) == len(g)):
        raise ValueError("paired arrays must have equal lengths")
    if len(np.unique(y)) < 2:
        return {"delta_auroc": None, "ci95": None, "valid_replicates": 0,
                "requested_replicates": repeats, "reason": "observed_labels_have_one_class"}
    observed = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    unique = np.unique(g)
    index = {group: np.flatnonzero(g == group) for group in unique}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index[group] for group in sampled])
        if len(np.unique(y[idx])) == 2:
            estimates.append(float(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx])))
    return {"delta_auroc": observed,
            "ci95": [float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))] if estimates else None,
            "valid_replicates": len(estimates), "requested_replicates": repeats,
            "resamples_with_both_classes": len(estimates), "unique_image_groups": int(len(unique))}


def grouped_bootstrap_mean(values, groups, repeats=1000, seed=17):
    """Bootstrap a mean over independent image groups, not question/variant rows."""
    import numpy as np
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    valid = np.isfinite(values)
    values, groups = values[valid], groups[valid]
    unique = np.unique(groups)
    if not len(unique):
        return None
    group_values = {group: float(np.mean(values[groups == group])) for group in unique}
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        estimates.append(float(np.mean([group_values[group] for group in sampled])))
    return {"mean": float(np.mean(estimates)),
            "ci95": [float(np.quantile(estimates, .025)), float(np.quantile(estimates, .975))],
            "groups": int(len(unique)), "valid_replicates": repeats}
