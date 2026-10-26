"""V2.5.5.3.3 reproducible A/B utility for *identical* feature matrices.
This is a model-level full-feature-array comparison, not a claim of raster I/O equivalence.
"""
import time
import numpy as np

def predict_in_batches(model, features, batch_size):
    features = np.asarray(features)
    if features.ndim != 2 or batch_size < 1:
        raise ValueError("Expected 2D features and positive batch size")
    outputs = []
    elapsed = 0.0
    for start in range(0, len(features), batch_size):
        t = time.perf_counter()
        outputs.append(model.predict_proba(features[start:start + batch_size]))
        elapsed += time.perf_counter() - t
    return np.concatenate(outputs, axis=0), elapsed

def compare_batches(model, features, batch_a=1024, batch_b=4096, tolerance=1e-12):
    """Use one fitted model and identical ordered pixels; serial execution is caller's responsibility."""
    a, ta = predict_in_batches(model, features, batch_a)
    b, tb = predict_in_batches(model, features, batch_b)
    diff = float(np.max(np.abs(a-b)))
    mismatches = int(np.count_nonzero(np.argmax(a, axis=1) != np.argmax(b, axis=1)))
    return {"pixels":len(features), "batch_a":batch_a, "batch_b":batch_b,
            "seconds_a":ta, "seconds_b":tb, "max_probability_difference":diff,
            "label_mismatches":mismatches, "pass":diff <= tolerance and mismatches == 0}
