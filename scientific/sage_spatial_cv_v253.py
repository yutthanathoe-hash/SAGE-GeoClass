# -*- coding: utf-8 -*-
"""SAGE-GeoClass V2.5.3.1 scientific-integrity spatial CV.
TRAIN-only GroupKFold, explicit class coverage and exact CNN patch non-overlap audit.
"""
import tempfile
import numpy as np
import importlib.util as _importlib_util
import os as _os
_optimizer_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'sage_fold_optimizer_v2532.py')
if not _os.path.isfile(_optimizer_path):
    raise ImportError('Missing sage_fold_optimizer_v2532.py alongside sage_spatial_cv_v253.py')
_optimizer_spec = _importlib_util.spec_from_file_location('sage_fold_optimizer_v2532', _optimizer_path)
_optimizer_module = _importlib_util.module_from_spec(_optimizer_spec)
_optimizer_spec.loader.exec_module(_optimizer_module)
optimize_spatial_folds = _optimizer_module.optimize_spatial_folds
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_recall_fscore_support
from scipy.spatial import cKDTree


def _patch_overlap_count(train, validation, rows, cols, radius):
    """Count training patch centers overlapping ANY validation patch (Chebyshev)."""
    if radius <= 0 or not len(train) or not len(validation):
        return 0
    centers = np.column_stack((np.asarray(rows)[validation], np.asarray(cols)[validation]))
    query = np.column_stack((np.asarray(rows)[train], np.asarray(cols)[train]))
    nearest = cKDTree(centers).query(query, k=1, p=np.inf)[0]
    return int(np.count_nonzero(nearest <= 2 * radius))


def run_spatial_cv(owner, key, display, X, X_patch, y, groups, rows, cols,
                   train_indices, n_classes, class_ids, n_folds, patch_size,
                   epochs, batch, model_dir, n_trees, max_depth, balance,
                   standardize, feedback, random_seed=42):
    idx = np.asarray(train_indices, dtype=np.int64)
    y = np.asarray(y)
    groups = np.asarray(groups)
    class_ids = np.asarray(class_ids)
    if n_folds < 2:
        raise ValueError('CV_FOLDS must be >= 2')
    if len(np.unique(groups[idx])) < n_folds:
        raise ValueError('Insufficient independent groups for %d folds' % n_folds)
    if key == 'SpatialCNN' and X_patch is None:
        raise ValueError('SpatialCNN requires extracted patches')
    if key == 'SpatialCNN' and patch_size % 2 != 1:
        raise ValueError('SpatialCNN patch size must be odd for center-based overlap audit')
    rows_out, classes_out, valid = [], [], []
    splits, fold_audit = optimize_spatial_folds(y, groups, idx, class_ids, n_folds, seed=random_seed)
    feedback.pushInfo('Class-aware spatial folds: %s; class group support=%s; missing class-fold pairs=%d' % (
        fold_audit['status'], fold_audit['class_group_support'], fold_audit['missing_class_fold_pairs']))
    if fold_audit['impossible_classes']:
        feedback.pushWarning('Full %d-fold class coverage mathematically impossible for classes %s; no spatial groups split' %
                             (n_folds, fold_audit['impossible_classes']))
    for fold, (tr_local, va_local) in enumerate(splits, 1):
        if feedback.isCanceled():
            raise RuntimeError('Spatial CV canceled')
        tr, va = idx[tr_local], idx[va_local]
        before = len(tr)
        purged = 0
        radius = patch_size // 2 if key == 'SpatialCNN' else 0
        overlap_before = _patch_overlap_count(tr, va, rows, cols, radius)
        if key == 'SpatialCNN':
            tr, purged = owner._purge_patch_overlap(tr, va, rows, cols, radius)
        overlap_after = _patch_overlap_count(tr, va, rows, cols, radius)
        group_overlap = int(len(np.intersect1d(groups[tr], groups[va])))
        if group_overlap or overlap_after:
            raise ValueError('Fold %d leakage: groups=%d patch_overlaps=%d' % (fold, group_overlap, overlap_after))
        if before - len(tr) != purged:
            raise ValueError('Fold %d purge count inconsistent' % fold)
        missing_train = sorted(set(class_ids.tolist()) - set(np.unique(y[tr]).tolist()))
        missing_val = sorted(set(class_ids.tolist()) - set(np.unique(y[va]).tolist()))
        row = dict(model=display, fold=fold, status='OK', n_train=len(tr),
                   n_validation=len(va), n_train_groups=len(np.unique(groups[tr])),
                   n_validation_groups=len(np.unique(groups[va])), purged_train=purged,
                   patch_overlap_before=overlap_before, patch_overlap_after=overlap_after,
                   group_overlap_after=group_overlap,
                   missing_train_classes=';'.join(map(str,missing_train)),
                   missing_validation_classes=';'.join(map(str,missing_val)),
                   reason='')
        if not len(tr) or not len(va) or missing_train or missing_val:
            row['status'] = 'INVALID_CLASS_COVERAGE'
            row['reason'] = ('missing train classes=%s; missing validation classes=%s' %
                             (missing_train, missing_val))
            rows_out.append(row)
            for cid in class_ids:
                support = int(np.sum(y[va] == cid))
                classes_out.append(dict(model=display,fold=fold,class_id=int(cid),support=support,
                                        precision='',recall='',f1='',status='INVALID_FOLD' if support else 'MISSING_CLASS'))
            feedback.pushWarning('Spatial CV fold %d INVALID: %s' % (fold,row['reason']))
            continue
        feedback.pushInfo('Spatial CV fold %d/%d: train=%d validation=%d purged=%d; patch overlap before=%d after=%d; class coverage=PASS' %
                          (fold,n_folds,len(tr),len(va),purged,overlap_before,overlap_after))
        with tempfile.TemporaryDirectory(prefix='sage_cv_fold_%02d_' % fold) as fold_dir:
            if key in ('SpatialCNN','TemporalCNN','DNN'):
                if key == 'SpatialCNN':
                    xtr, xva = X_patch[tr], X_patch[va]
                elif key == 'TemporalCNN':
                    xtr, xva = X[tr], X[va]
                else:
                    xtr, xva = X[tr].reshape(len(tr),-1), X[va].reshape(len(va),-1)
                    if standardize:
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler().fit(xtr)
                        xtr, xva = scaler.transform(xtr), scaler.transform(xva)
                model, pred = owner._train_tf_isolated(key,xtr,y[tr],xva,y[va],n_classes,epochs,batch,fold_dir,feedback)
                owner._tf_models.remove(model)
                model.close()
                owner._active_tf_model = None
            else:
                model = owner._make_ml_model(key,n_trees,max_depth,balance,standardize)
                xtr, xva = X[tr].reshape(len(tr),-1), X[va].reshape(len(va),-1)
                top_k = getattr(owner, '_sage_feature_top_k', 0)
                if top_k:
                    import importlib.util as _fi_util
                    _fi_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'sage_model_feature_v254.py')
                    _fi_spec = _fi_util.spec_from_file_location('sage_model_feature_v254', _fi_path)
                    _fi_module = _fi_util.module_from_spec(_fi_spec)
                    _fi_spec.loader.exec_module(_fi_module)
                    wrap_ml_selector = _fi_module.wrap_ml_selector
                    model = wrap_ml_selector(model, top_k, getattr(owner, '_sage_feature_seed', 42))
                model.fit(xtr,y[tr]); pred=model.predict(xva)
        truth=y[va]
        row.update(accuracy=float(accuracy_score(truth,pred)),
                   macro_f1=float(f1_score(truth,pred,labels=class_ids,average='macro',zero_division=0)),
                   balanced_accuracy=float(recall_score(truth,pred,labels=class_ids,average='macro',zero_division=0)))
        rows_out.append(row);valid.append(row)
        p,r,f,s=precision_recall_fscore_support(truth,pred,labels=class_ids,zero_division=0)
        for cid,pr,rc,f1,sup in zip(class_ids,p,r,f,s):
            classes_out.append(dict(model=display,fold=fold,class_id=int(cid),support=int(sup),
                                    precision=float(pr),recall=float(rc),f1=float(f1),status='OK'))
    summary={'fold_optimization':fold_audit, 'status':'PASS' if len(valid)==n_folds else 'INCOMPLETE_CLASS_COVERAGE',
             'completed':len(valid), 'reason':'' if len(valid)==n_folds else '%d/%d valid folds; statistics exclude invalid folds' % (len(valid),n_folds)}
    for metric in ('accuracy','macro_f1','balanced_accuracy'):
        values=np.asarray([r[metric] for r in valid],dtype=float)
        summary[metric+'_mean']=float(values.mean()) if len(values) else float('nan')
        summary[metric+'_std']=float(values.std(ddof=1)) if len(values)>1 else float('nan')
    return rows_out,classes_out,summary
