# -*- coding: utf-8 -*-
"""Train-only spatial feature ablation audit for pixel ML; never accesses holdout test."""
import csv
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.base import clone
from sage_fold_optimizer_v2532 import optimize_spatial_folds

FIELDS = ['experiment','feature','fold','status','n_train','n_validation','macro_f1','delta_vs_full','reason']

def audit_ml(owner, key, X, y, groups, train_indices, class_ids, n_folds,
             names, n_trees, max_depth, balance, standardize, seed=42,
             top_k=0, max_features=20, feedback=None):
    """Each fold fits new model and selector on its training subset only.
    Negative delta_vs_full indicates removing feature improved performance.
    """
    if key in ('SpatialCNN','TemporalCNN','DNN'):
        raise ValueError('CNN/DNN retraining ablation is not implemented; ML-only audit')
    idx = np.asarray(train_indices, dtype=np.int64)
    y = np.asarray(y); X = np.asarray(X).reshape(len(y), -1)
    if len(names) != X.shape[1]:
        names = ['Feature_%d' % (j+1) for j in range(X.shape[1])]
    if X.shape[1] > max_features:
        raise ValueError('Too many features for exhaustive ablation; increase audit limit intentionally')
    splits, audit = optimize_spatial_folds(y, np.asarray(groups), idx, np.asarray(class_ids), n_folds, seed=seed)
    results = []
    labels = np.asarray(class_ids)
    for fold,(a,b) in enumerate(splits, 1):
        tr,va = idx[a],idx[b]
        missing = set(labels.tolist())-set(np.unique(y[tr]).tolist())
        missing |= set(labels.tolist())-set(np.unique(y[va]).tolist())
        if missing:
            results.append(dict(experiment='fold',feature='',fold=fold,status='INVALID_CLASS_COVERAGE',reason=str(sorted(missing))))
            continue
        def score(cols, selector_k=0):
            model = owner._make_ml_model(key,n_trees,max_depth,balance,standardize)
            if selector_k and selector_k < len(cols):
                from sage_model_feature_v254 import wrap_ml_selector
                model = wrap_ml_selector(model,selector_k,seed)
            model.fit(X[np.ix_(tr,cols)],y[tr]); p=model.predict(X[np.ix_(va,cols)])
            return float(f1_score(y[va],p,labels=labels,average='macro',zero_division=0))
        allcols = list(range(X.shape[1]))
        baseline = score(allcols)
        results.append(dict(experiment='full',feature='ALL',fold=fold,status='PASS',n_train=len(tr),n_validation=len(va),macro_f1=baseline,delta_vs_full=0,reason=''))
        for j,name in enumerate(names):
            cols = [c for c in allcols if c != j]
            if not cols: continue
            val = score(cols)
            results.append(dict(experiment='leave_one_out',feature=name,fold=fold,status='PASS',n_train=len(tr),n_validation=len(va),macro_f1=val,delta_vs_full=baseline-val,reason=''))
        if top_k and top_k < len(allcols):
            val = score(allcols,top_k)
            results.append(dict(experiment='top_k_%d'%top_k,feature='TRAIN_FOLD_SELECTED',fold=fold,status='PASS',n_train=len(tr),n_validation=len(va),macro_f1=val,delta_vs_full=baseline-val,reason=''))
        if feedback: feedback.pushInfo('V2.5.4.1 ML ablation fold %d/%d complete' % (fold,n_folds))
    return results, audit

def write_audit(path, rows):
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS,extrasaction='ignore');w.writeheader();w.writerows(rows)
