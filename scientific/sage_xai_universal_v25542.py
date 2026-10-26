"""SAGE-GeoClass v2.5.5.4.2: optional, read-only model-aware XAI diagnostics.
Only pixel-based sklearn models are supported; deep networks are explicitly NOT_RUN.
"""
import csv
import numpy as np

def fit_ood_reference(x, seed=42, cap=30000):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or len(x)<2 or not np.isfinite(x).all():
        raise ValueError('OOD reference requires finite 2D TRAIN features')
    if len(x)>cap:
        x=x[np.random.default_rng(seed).choice(len(x),cap,replace=False)]
    center=np.median(x,axis=0)
    scale=np.maximum(np.percentile(x,75,axis=0)-np.percentile(x,25,axis=0),1e-8)
    dist=ood_scores(x,center,scale)
    return center,scale,float(np.quantile(dist,.99))

def ood_scores(x,center,scale):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=len(center):raise ValueError('OOD feature schema mismatch')
    return np.sqrt(np.mean(((x-center)/scale)**2,axis=1)).astype(np.float32)

def reliability(confidence,ood,threshold):
    confidence=np.asarray(confidence,dtype=np.float32)
    ood=np.asarray(ood,dtype=np.float32)
    penalty=np.maximum(0.,ood/max(float(threshold),1e-8)-1.)
    return np.clip(confidence/(1.+penalty),0.,1.).astype(np.float32)

def _write_status(output,status,reason):
    with open(output,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['status','reason']);w.writerow([status,reason])

def spatial_group_importance(model,x,y,groups,names,output,seed=42,cap=4000,repeats=3):
    """Local group baseline; group-weighted permutation loss, held-out only."""
    from sklearn.metrics import accuracy_score
    x=np.asarray(x);y=np.asarray(y);groups=np.asarray(groups)
    if x.ndim!=2 or x.shape[1]!=len(names) or not (len(x)==len(y)==len(groups)):
        raise ValueError('Spatial importance feature/group shape mismatch')
    rng=np.random.default_rng(seed)
    ids=np.arange(len(x))
    if len(ids)>cap:ids=np.sort(rng.choice(ids,cap,replace=False))
    x=x[ids];y=y[ids];groups=groups[ids]
    rows=[]
    for j,name in enumerate(names):
        drops=[];weights=[]
        for g in np.unique(groups):
            ix=np.flatnonzero(groups==g)
            if len(ix)<2:continue
            local_x=x[ix];local_y=y[ix]
            base=accuracy_score(local_y,model.predict(local_x))
            for _ in range(repeats):
                changed=local_x.copy();changed[:,j]=local_x[rng.permutation(len(ix)),j]
                drops.append(base-accuracy_score(local_y,model.predict(changed)))
                weights.append(len(ix))
        rows.append([name,float(accuracy_score(y,model.predict(x))),
                     float(np.average(drops,weights=weights)) if drops else float('nan'),
                     len(np.unique(groups)),len(drops),len(x)])
    with open(output,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['feature','baseline_accuracy','mean_group_permutation_drop','groups_total','permutations_evaluated','sample_count']);w.writerows(rows)
    return rows

def shap_summary(model,x,names,output,model_key='RF',seed=42,cap=200, sample_output=None, classwise_output=None):
    """TreeSHAP for RF/ET/HistGB/XGBoost; bounded KernelSHAP for SVM/MLP and fallback.
    Optional shap dependency. Failures are propagated and caller records FAILED.
    """
    import shap
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[1]!=len(names):raise ValueError('SHAP feature schema mismatch')
    rng=np.random.default_rng(seed)
    if model_key in ('SVM','MLP','LightGBM','CatBoost','LogReg'):
        if not hasattr(model,'predict_proba'):
            _write_status(output,'NOT_RUN','Model has no predict_proba for KernelSHAP');return None
        foreground=x[rng.choice(len(x),min(len(x),cap),replace=False)]
        background=x[rng.choice(len(x),min(len(x),32),replace=False)]
        explainer=shap.KernelExplainer(model.predict_proba,background)
        vals=explainer.shap_values(foreground,nsamples=max(64,2*x.shape[1]+32),silent=True)
        method='KernelSHAP'
    else:
        foreground=x[rng.choice(len(x),min(len(x),cap),replace=False)]
        if model_key == 'XGBoost':
            try:
                # Unwrap label-encoding adapter; probability columns retain original class order.
                explainer=shap.TreeExplainer(model._xgb_model)
                vals=explainer.shap_values(foreground,check_additivity=False)
                method='TreeSHAP (XGBoost raw margin)'
            except Exception:
                # SHAP / XGBoost versions can be incompatible; bounded probability-space fallback.
                background=x[rng.choice(len(x),min(len(x),32),replace=False)]
                explainer=shap.KernelExplainer(model.predict_proba,background)
                vals=explainer.shap_values(foreground,nsamples=max(64,2*x.shape[1]+32),silent=True)
                method='KernelSHAP (XGBoost probability fallback)'
        else:
            explainer=shap.TreeExplainer(model)
            vals=explainer.shap_values(foreground,check_additivity=False)
            method='TreeSHAP'
    # Preserve signed per-sample/per-class contributions; do not average
    # before writing beeswarm data. Normalize SHAP API variants to N,F,K.
    if isinstance(vals,list):
        signed=np.stack([np.asarray(v) for v in vals],axis=-1)
    else:
        signed=np.asarray(vals)
        if signed.ndim==2:signed=signed[:,:,None]
    if signed.ndim!=3 or signed.shape[:2]!=(len(foreground),len(names)):
        raise ValueError('Unexpected SHAP sample shape: %s'%(signed.shape,))
    if not np.isfinite(signed).all():raise ValueError('Nonfinite SHAP values')
    import importlib.util as _ilu, os as _os
    _vp=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "sage_shap_visualization_v255432.py")
    _vs=_ilu.spec_from_file_location("sage_shap_visualization_v255432",_vp)
    _vm=_ilu.module_from_spec(_vs);_vs.loader.exec_module(_vm)
    write_samples,write_classwise=_vm.write_samples,_vm.write_classwise
    classes=list(getattr(model,'classes_',range(signed.shape[-1])))
    if len(classes)!=signed.shape[-1]:classes=list(range(signed.shape[-1]))
    if sample_output:write_samples(sample_output,foreground,signed,names,classes,model_key,'FEATURE_VALUE')
    if classwise_output:write_classwise(classwise_output,signed,names,classes,model_key)
    if isinstance(vals,list):
        perclass=np.stack([np.abs(np.asarray(v)).mean(axis=0) for v in vals],axis=0)
        importance=perclass.mean(axis=0)
    else:
        arr=np.asarray(vals)
        if arr.ndim==2:importance=np.abs(arr).mean(axis=0)
        elif arr.ndim==3:importance=np.abs(arr).mean(axis=(0,2))
        else:raise ValueError('Unexpected SHAP output dimensions')
    if importance.shape!=(len(names),):raise ValueError('SHAP feature count mismatch')
    with open(output,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['feature','mean_absolute_shap','method','sample_count','model']);
        w.writerows((name,float(v),method,len(foreground),model_key) for name,v in zip(names,importance))
    return importance
