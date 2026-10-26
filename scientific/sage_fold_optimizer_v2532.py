# -*- coding: utf-8 -*-
"""Class-aware spatial group fold assignment; never splits a spatial group."""
import numpy as np
from sklearn.model_selection import GroupKFold


def optimize_spatial_folds(y, groups, indices, class_ids, n_folds, seed=42, attempts=256):
    """Return (train,validation) local-index splits, audit. Deterministic bounded search.

    Objective prioritizes complete class coverage, then class/count balance.
    Infeasibility is reported, not hidden by splitting groups.
    """
    idx = np.asarray(indices, dtype=np.int64)
    yy = np.asarray(y)[idx]
    gg = np.asarray(groups)[idx]
    labels = np.asarray(class_ids)
    uniq, inv = np.unique(gg, return_inverse=True)
    g, k, c = len(uniq), int(n_folds), len(labels)
    if k < 2 or g < k:
        raise ValueError('Spatial CV requires at least n_folds distinct spatial groups')
    counts = np.zeros((g, c), dtype=np.int64)
    for j, cid in enumerate(labels):
        counts[:, j] = np.bincount(inv, weights=(yy == cid).astype(np.int64), minlength=g).astype(np.int64)
    presence = (counts > 0).sum(axis=0)
    impossible = [int(labels[j]) for j in range(c) if presence[j] < k]
    sizes = counts.sum(axis=1)
    total = counts.sum(axis=0)
    target = total / k
    rng = np.random.RandomState(int(seed))

    def score(assignment):
        fold_counts = np.zeros((k, c), dtype=np.int64)
        for f in range(k):
            fold_counts[f] = counts[assignment == f].sum(axis=0)
        missing = int(np.count_nonzero(fold_counts == 0))
        empty = int(np.count_nonzero(fold_counts.sum(axis=1) == 0))
        # Prioritize coverage absolutely; penalize class imbalance relative to class prevalence.
        relative = (fold_counts - target) / np.maximum(target, 1)
        cost = float((relative ** 2).sum() + 0.1 * np.var(fold_counts.sum(axis=1) / max(sizes.sum()/k, 1)))
        return (empty, missing, cost), fold_counts

    best = None
    # GroupKFold is always a safe fallback if optimization cannot improve it.
    baseline = np.empty(g, dtype=np.int32)
    for f, (_, va) in enumerate(GroupKFold(n_splits=k).split(idx, yy, gg)):
        baseline[np.unique(inv[va])] = f
    best_score, best_counts = score(baseline)
    best = baseline.copy()
    # Greedy group allocation, with rarity-first order and randomized restarts.
    rarity = (counts > 0) @ (1.0 / np.maximum(presence, 1))
    for attempt in range(max(1, int(attempts))):
        order = np.lexsort((rng.random(g), -sizes, -rarity))
        assignment = np.full(g, -1, dtype=np.int32)
        fc = np.zeros((k, c), dtype=np.int64)
        for pos, group in enumerate(order):
            if pos < k:
                f = pos
            else:
                costs = []
                for f0 in range(k):
                    new = fc.copy(); new[f0] += counts[group]
                    # Reward covering classes in new fold, penalize excess concentrations.
                    miss = np.count_nonzero(new == 0)
                    imbalance = np.sum(((new - target) / np.maximum(target, 1))**2)
                    costs.append((miss * 100 + imbalance, rng.random(), f0))
                f = min(costs)[2]
            assignment[group] = f
            fc[f] += counts[group]
        # Local group moves and swaps to improve class coverage, preserving nonempty folds.
        current, _ = score(assignment)
        for _ in range(3):
            changed = False
            for group in rng.permutation(g):
                source = assignment[group]
                if np.count_nonzero(assignment == source) <= 1:
                    continue
                for dest in rng.permutation(k):
                    if dest == source: continue
                    assignment[group] = dest
                    candidate, _ = score(assignment)
                    if candidate < current:
                        current = candidate; changed = True; source = dest
                    else:
                        assignment[group] = source
            if not changed: break
        current, current_counts = score(assignment)
        if current < best_score:
            best_score, best_counts, best = current, current_counts, assignment.copy()
        if best_score[0] == 0 and best_score[1] == 0 and best_score[2] < 0.5:
            break
    splits = []
    for f in range(k):
        va = np.flatnonzero(best[inv] == f)
        tr = np.flatnonzero(best[inv] != f)
        splits.append((tr, va))
    audit = {'status': 'CLASS_COVERAGE_FEASIBLE' if not impossible else 'CLASS_COVERAGE_IMPOSSIBLE_BY_GROUP_SUPPORT',
             'impossible_classes': impossible,
             'class_group_support': {int(labels[j]): int(presence[j]) for j in range(c)},
             'missing_class_fold_pairs': int(best_score[1]),
             'all_folds_covered_before_purge': bool(best_score[1] == 0),
             'fold_class_counts': best_counts.tolist(),
             'n_spatial_groups': g, 'n_folds': k, 'seed': int(seed)}
    return splits, audit
