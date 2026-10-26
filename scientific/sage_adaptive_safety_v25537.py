"""Read-only safety audit for adaptive ML inference; no changes to model output."""
import math

def audit_available(available_bytes, reserve_bytes):
    """Return availability state and headroom; invalid samples are not treated as zero."""
    if available_bytes is None:
        return {'status': 'unavailable', 'headroom_bytes': None}
    if isinstance(available_bytes, bool) or not isinstance(available_bytes, (int, float)):
        return {'status': 'invalid', 'headroom_bytes': None}
    if not math.isfinite(available_bytes) or available_bytes < 0:
        return {'status': 'invalid', 'headroom_bytes': None}
    headroom = int(available_bytes) - int(reserve_bytes)
    return {'status': 'below_reserve' if headroom <= 0 else 'ok', 'headroom_bytes': headroom}
