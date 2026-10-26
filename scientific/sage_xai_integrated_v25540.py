"""Optional explainability diagnostics for SAGE-GeoClass; no model mutation."""
import csv
import numpy as np

def fit_ood_reference(x, seed=42, cap=30000):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Training features must be a finite 2D matrix')
    if len(x) > cap:
        x = x[np.random.default_rng(seed).choice(len(x), cap, replace=False)]
    center = np.median(x, axis=0)
    scale = np.maximum(np.percentile(x, 75, axis=0)-np.percentile(x, 25, axis=0), 1e-8)
    dist = np.sqrt(np.mean(((x-center)/scale)**2, axis=1))
    threshold = float(np.quantile(dist, .99))
    return center, scale, threshold

def ood_scores(x, center, scale):
    x = np.asarray(x, dtype=np.float64)
    return np.sqrt(np.mean(((x-center)/scale)**2, axis=1)).astype(np.float32)

def reliability(confidence, ood, threshold):
    """Heuristic reliability, not calibrated correctness probability."""
    confidence = np.asarray(confidence, dtype=np.float32)
    ood = np.asarray(ood, dtype=np.float32)
    penalty = np.maximum(0., ood / max(float(threshold), 1e-8) - 1.)
    return (confidence / (1. + penalty)).astype(np.float32)

def spatial_group_importance(model, x, y, groups, names, output, seed=42, cap=4000):
    """Group-level permutation diagnostic on held-out validation only."""
    from sklearn.metrics import accuracy_score
    x=np.asarray(x); y=np.asarray(y); groups=np.asarray(groups)
    if not (len(x)==len(y)==len(groups)): raise ValueError('Length mismatch')
    rng=np.random.default_rng(seed)
    ids=np.arange(len(x))
    if len(ids)>cap: ids=np.sort(rng.choice(ids, cap, replace=False))
    x=x[ids];y=y[ids];groups=groups[ids]
    base=accuracy_score(y,model.predict(x))
    rows=[]
    for j,name in enumerate(names):
        scores=[]
        for g in np.unique(groups):
            ix=np.flatnonzero(groups==g)
            if len(ix)<2:continue
            copy=x.copy();copy[ix,j]=x[rng.permutation(ix),j]
            scores.append((len(ix),base-accuracy_score(y,model.predict(copy))))
        rows.append([name,base,float(np.average([v for _,v in scores],weights=[w for w,_ in scores])) if scores else float('nan'),len(scores)])
    with open(output,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['feature','baseline_accuracy','mean_group_permutation_drop','groups_evaluated']);w.writerows(rows)
    return rows

def shap_summary(model, x, names, output, seed=42, cap=250):
    """Optional TreeExplainer; unsupported models report NOT_RUN at caller."""
    import shap
    x=np.asarray(x)
    if len(x)>cap:x=x[np.random.default_rng(seed).choice(len(x),cap,replace=False)]
    explainer=shap.TreeExplainer(model)
    values=explainer.shap_values(x,check_additivity=False)
    if isinstance(values,list):
        importance=np.mean([np.abs(v).mean(axis=0) for v in values],axis=0)
    else:
        arr=np.asarray(values)
        importance=np.abs(arr).mean(axis=tuple(i for i in range(arr.ndim) if i!=1))
    if len(importance)!=len(names):raise ValueError('SHAP feature count mismatch')
    with open(output,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['feature','mean_absolute_shap']);w.writerows(zip(names,importance))
    return importance
