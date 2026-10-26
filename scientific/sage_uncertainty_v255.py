"""SAGE-GeoClass V2.5.5 uncertainty utilities; no QGIS dependency."""
import csv
import numpy as np


def normalized_probabilities(probs, n_classes):
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != n_classes:
        raise ValueError('Expected probability matrix (samples, %d classes)' % n_classes)
    if not np.all(np.isfinite(p)) or np.any(p < -1e-7):
        raise ValueError('Invalid probabilities')
    p = np.clip(p, 0.0, None)
    sums = p.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError('Probability rows must have positive sum')
    return p / sums


def uncertainty(probs, n_classes):
    p = normalized_probabilities(probs, n_classes)
    top = np.max(p, axis=1)
    if n_classes > 1:
        top2 = np.partition(p, -2, axis=1)[:, -2]
        margin = top - top2
        entropy = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-15)), 0), axis=1) / np.log(n_classes)
    else:
        margin = np.ones(len(p))
        entropy = np.zeros(len(p))
    return top.astype('float32'), entropy.astype('float32'), margin.astype('float32')


def reliability_rows(probs, y_true, class_ids, bins=10):
    p = normalized_probabilities(probs, len(class_ids))
    truth = np.asarray(y_true)
    confidence, entropy, margin = uncertainty(p, len(class_ids))
    predicted = np.asarray(class_ids)[np.argmax(p, axis=1)]
    correct = (predicted == truth).astype(float)
    edges = np.linspace(0, 1, bins+1)
    rows = []
    ece = 0.0
    for b in range(bins):
        mask = (confidence >= edges[b]) & (confidence < edges[b+1] if b < bins-1 else confidence <= edges[b+1])
        count = int(mask.sum())
        accuracy = float(correct[mask].mean()) if count else float('nan')
        mean_conf = float(confidence[mask].mean()) if count else float('nan')
        if count: ece += count / len(confidence) * abs(accuracy - mean_conf)
        rows.append({'bin': b+1, 'lower': edges[b], 'upper': edges[b+1], 'n': count,
                     'accuracy': accuracy, 'mean_confidence': mean_conf, 'gap': abs(accuracy-mean_conf) if count else float('nan')})
    truth_onehot = (truth[:, None] == np.asarray(class_ids)[None, :]).astype(float)
    summary = {'n_test': len(truth), 'accuracy': float(correct.mean()),
               'mean_confidence': float(confidence.mean()), 'mean_entropy': float(entropy.mean()),
               'mean_margin': float(margin.mean()), 'ece_10bin': float(ece),
               'multiclass_brier': float(np.mean(np.sum((p-truth_onehot)**2, axis=1))),
               'note': 'Independent-test diagnostics only; uncalibrated scores; no test-based tuning'}
    return rows, summary


def write_rows(path, rows):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['status'])
        w.writeheader()
        w.writerows(rows)
