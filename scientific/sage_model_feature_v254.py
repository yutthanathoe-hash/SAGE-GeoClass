# -*- coding: utf-8 -*-
"""SAGE-GeoClass V2.5.4: leakage-conscious model and feature diagnostics.
Feature selector is fitted within each ML training partition, including CV folds.
Permutation importance is diagnostic on the model-selection validation partition.
"""
import csv
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score


def wrap_ml_selector(estimator, top_k, seed=42):
    if top_k <= 0:
        return estimator
    selector = SelectFromModel(ExtraTreesClassifier(n_estimators=128, random_state=seed,
                        n_jobs=-1, class_weight='balanced'), threshold=-np.inf,
                        max_features=int(top_k))
    return Pipeline([('sage_feature_selector', selector), ('sage_classifier', estimator)])


def selected_mask(model, n_features):
    if hasattr(model, 'named_steps') and 'sage_feature_selector' in model.named_steps:
        return model.named_steps['sage_feature_selector'].get_support().astype(bool)
    return np.ones(n_features, dtype=bool)


def _predict(model, key, x, scaler=None):
    if key in ('SpatialCNN', 'TemporalCNN', 'DNN'):
        if key == 'DNN' and scaler is not None:
            x = scaler.transform(x)
        return np.argmax(model.predict(x, verbose=0), axis=1) + 1
    return model.predict(x)


def permutation_feature_importance(model, key, x, y, feature_names, n_classes,
                                   repeats=5, seed=42, scaler=None, max_samples=2500):
    """Permutation of entire channels, not individual pixels in a CNN patch.
    Uses macro F1 and fixed labels; does not use independent test labels.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(y) > max_samples:
        ids = np.random.default_rng(seed).choice(len(y), max_samples, replace=False)
        x, y = x[ids], y[ids]
    if len(y) == 0:
        raise ValueError('No validation samples for importance')
    labels = np.arange(1, n_classes + 1)
    base = float(f1_score(y, _predict(model, key, x, scaler),
                          labels=labels, average='macro', zero_division=0))
    rng = np.random.default_rng(seed)
    rows = []
    if key == 'SpatialCNN':
        n_features = x.shape[-1]
    elif key == 'TemporalCNN':
        n_features = x.shape[-1]  # channel at every timestep jointly
    else:
        n_features = x.shape[1]
    if len(feature_names) != n_features:
        feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    mask = selected_mask(model, n_features) if key not in ('SpatialCNN','TemporalCNN','DNN') else np.ones(n_features,bool)
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(repeats):
            shuffled = x.copy()
            order = rng.permutation(len(y))
            if key in ('SpatialCNN', 'TemporalCNN'):
                shuffled[..., j] = x[order, ..., j]
            else:
                shuffled[:, j] = x[order, j]
            perm = float(f1_score(y, _predict(model, key, shuffled, scaler),
                                  labels=labels, average='macro', zero_division=0))
            drops.append(base - perm)
        rows.append({'feature':name, 'feature_index':j+1, 'selected_for_training':int(mask[j]),
                     'importance_mean':float(np.mean(drops)),
                     'importance_std':float(np.std(drops,ddof=1)) if len(drops)>1 else 0.,
                     'baseline_macro_f1':base, 'method':'validation_permutation_macro_f1',
                     'note':'post_selection_diagnostic_not_unbiased_test'})
    rows.sort(key=lambda r:(-r['importance_mean'],r['feature_index']))
    for rank,row in enumerate(rows,1): row['rank']=rank
    return rows


def write_importance(path, rows):
    fields=['rank','feature','feature_index','selected_for_training','importance_mean',
            'importance_std','baseline_macro_f1','method','note']
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
