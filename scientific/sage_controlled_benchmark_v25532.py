"""Opt-in paired RF inference diagnostics, independent of QGIS and training.
Only benchmark a small fixed raster sample. Do not alter model weights.
"""
import csv
import time
import numpy as np

COLUMNS = ['status','sequence','batch_size','adaptive','sample_pixels','batches',
           'prediction_seconds','preparation_seconds','memory_seconds',
           'wall_seconds','pixels_per_prediction_second','max_abs_probability_difference',
           'label_mismatches','memory_checks','downshifts','peak_rss_bytes','note']


def write_results(path, rows):
    with open(path, 'w', newline='', encoding='utf-8-sig') as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)


def write_skipped(path, reason):
    write_results(path, [{'status': 'SKIPPED', 'note': reason}])


def _predict(model, features, batch_size, adaptive, monitor, reserve_bytes):
    # Pure numerical guard, mirrored from the frozen memory helper; no sys.path assumptions.
    def adaptive_batch(current, maximum, available_bytes, reserve_bytes, feature_count=4):
        current = max(256, min(int(current), int(maximum)))
        if available_bytes is None: return current
        headroom = int(available_bytes) - int(reserve_bytes)
        if headroom <= 0: return 256
        estimated = current * max(1, int(feature_count)) * 8 * 64
        while current > 256 and estimated > headroom // 4:
            current = max(256, current // 2)
            estimated = current * max(1, int(feature_count)) * 8 * 64
        return current
    pieces = []
    pred_time = prep_time = mem_time = 0.0
    batches = checks = downshifts = 0
    peak_rss = 0
    step = batch_size
    start = time.perf_counter()
    offset = 0
    while offset < len(features):
        if batches % 32 == 0:
            t = time.perf_counter()
            snap = monitor.sample()
            mem_time += time.perf_counter() - t
            checks += 1
            peak_rss = max(peak_rss, snap.get('rss_bytes') or 0)
            if adaptive and monitor.available_supported:
                updated = adaptive_batch(step, batch_size, snap.get('available_bytes'), reserve_bytes, features.shape[-1])
                if updated < step:
                    downshifts += 1
                step = updated
        t = time.perf_counter()
        part = features[offset:offset+step].reshape(-1, features.shape[-1])
        prep_time += time.perf_counter() - t
        t = time.perf_counter()
        proba = model.predict_proba(part)
        pred_time += time.perf_counter() - t
        pieces.append(proba)
        offset += len(part)
        batches += 1
    wall = time.perf_counter() - start
    return np.concatenate(pieces, axis=0), dict(batches=batches, prediction_seconds=pred_time,
        preparation_seconds=prep_time, memory_seconds=mem_time, wall_seconds=wall,
        memory_checks=checks, downshifts=downshifts, peak_rss_bytes=peak_rss)


def run_paired_benchmark(model, features, monitor, reserve_bytes, feedback=None):
    """Counterbalanced 2x2 paired repeated design on identical fixed pixels.

    Runs 8 combinations in forward/reverse order. Results are diagnostic, not
    a controlled system-wide benchmark; background load and thermal state remain.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or len(features) == 0:
        raise ValueError('Expected nonempty 2D features')
    if not hasattr(model, 'predict_proba'):
        raise ValueError('Model lacks predict_proba')
    settings = [(1024, False), (1024, True), (4096, True), (4096, False)]
    order = settings + list(reversed(settings))
    reference = None
    rows = []
    old_jobs = getattr(model, 'n_jobs', None)
    if old_jobs is not None:
        model.n_jobs = 1
    try:
        for ix, (size, adaptive) in enumerate(order, 1):
            probs, data = _predict(model, features, size, adaptive, monitor, reserve_bytes)
            if reference is None:
                reference = probs.copy()
            diff = float(np.max(np.abs(probs - reference)))
            mismatches = int(np.count_nonzero(np.argmax(probs, axis=1) != np.argmax(reference, axis=1)))
            row = dict(status='PASS' if diff <= 1e-12 and mismatches == 0 else 'NUMERICAL_MISMATCH',
                       sequence=ix, batch_size=size, adaptive=int(adaptive),
                       sample_pixels=len(features), max_abs_probability_difference=diff,
                       label_mismatches=mismatches, note='fixed sample; repeated reversed order', **data)
            row['pixels_per_prediction_second'] = round(len(features) / data['prediction_seconds'], 3) if data['prediction_seconds'] else 0
            rows.append(row)
            if feedback is not None:
                feedback.pushInfo('Diagnostic %d/8: batch=%d adaptive=%s prediction=%.3fs diff=%.3g' % (
                    ix, size, adaptive, data['prediction_seconds'], diff))
    finally:
        if old_jobs is not None:
            model.n_jobs = old_jobs
    return rows
