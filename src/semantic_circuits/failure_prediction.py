"""Leakage-aware training/evaluation utilities for SCC failure prediction."""
from __future__ import annotations


DEFAULT_FEATURES = ["answer_confidence", "clip_region_cosine", "grounding_score",
                    "external_necessity_js", "external_necessity_js_control_adjusted",
                    "external_sufficiency", "cps_weighted",
                    "ceca_distribution", "scc"]


def flatten_pathway_features(result: dict, failure_label=None, sample_id=None) -> dict:
    """Flatten one runner record into a failure-prediction row; optional external scores stay null."""
    probabilities=result.get("baseline_candidate_distribution") or {}
    external=result.get("external_intervention") or {}
    proposals=(result.get("data_provenance") or {}).get("evidence_proposals") or []
    grounding=[item.get("grounding_score") for item in proposals if item.get("grounding_score") is not None]
    clip=[item.get("region_clip_cosine") for item in proposals if item.get("region_clip_cosine") is not None]
    row={"sample_id":sample_id or result.get("sample_id"),
         "answer_confidence":max(probabilities.values()) if probabilities else None,
         "grounding_score":max(grounding) if grounding else None,
         "clip_region_cosine":max(clip) if clip else None,
         "external_necessity_js":external.get("necessity_js"),
         "external_necessity_js_control_adjusted":external.get("necessity_js_control_adjusted"),
         "external_sufficiency":external.get("sufficiency_similarity"),
         "cps_weighted":result.get("cps_weighted"),
         "ceca_distribution":result.get("ceca_distribution_mean"),
         "scc":result.get("scc")}
    if failure_label is not None: row["future_failure"]=int(failure_label)
    return row


def fit_failure_predictor(train_rows: list[dict], feature_columns=None,
                          target_column="future_failure", group_column="sample_id"):
    """Fit a train-only logistic baseline; caller must pass training rows only."""
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    feature_columns = feature_columns or DEFAULT_FEATURES
    rows=[r for r in train_rows if r.get(target_column) is not None]
    if not rows: raise ValueError("No labeled training rows")
    X=np.asarray([[r.get(c, np.nan) for c in feature_columns] for r in rows],dtype=float)
    y=np.asarray([int(r[target_column]) for r in rows])
    if len(np.unique(y))<2: raise ValueError("Training requires both failure and non-failure examples")
    model=Pipeline([("impute",SimpleImputer(strategy="median",add_indicator=True)),
                    ("scale",StandardScaler()),
                    ("logistic",LogisticRegression(class_weight="balanced",max_iter=2000,random_state=17))])
    model.fit(X,y)
    metadata={"features":feature_columns,"target":target_column,"group_column":group_column,
              "n_train":len(rows),"train_sample_ids":sorted({str(r.get(group_column)) for r in rows})}
    return model,metadata


def score_failure_predictor(model, rows: list[dict], feature_columns=None):
    import numpy as np
    feature_columns=feature_columns or DEFAULT_FEATURES
    X=np.asarray([[r.get(c,np.nan) for c in feature_columns] for r in rows],dtype=float)
    return model.predict_proba(X)[:,1].tolist()


def validation_threshold(labels, scores, objective="youden") -> float:
    """Choose threshold on validation only; caller must not pass test rows."""
    import numpy as np
    from sklearn.metrics import roc_curve
    fpr,tpr,thresholds=roc_curve(np.asarray(labels,dtype=int),np.asarray(scores,dtype=float))
    if objective=="youden": return float(thresholds[int(np.argmax(tpr-fpr))])
    raise ValueError("Supported objective: youden")
