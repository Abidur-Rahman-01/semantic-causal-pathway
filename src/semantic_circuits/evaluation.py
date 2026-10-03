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
    if not estimates: return None
    return {"mean":float(np.mean(estimates)),"ci95":[float(np.quantile(estimates,.025)),float(np.quantile(estimates,.975))],"valid_replicates":len(estimates)}
