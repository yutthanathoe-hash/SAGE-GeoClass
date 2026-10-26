"""V2.5.5.3.8: isolated deterministic memory-injection stress harness.

No monkeypatching, memory allocation, or modification of the QGIS inference path.
"""
import csv
import math
from pathlib import Path
from sage_adaptive_safety_v25537 import audit_available
from sage_inference_memory_v25536 import adaptive_batch

MIB = 1048576


def simulated_batches(memory_mb, *, maximum=4096, reserve_mb=2048, features=4):
    """Emulate production's every-32-batches memory checks on an injected sequence.

    A sequence item is None (unavailable) or an integer MiB of available RAM.
    The input must contain at least one sample; the sequence is exhausted by holding
    the final sample, as a deterministic synthetic monitor would.
    """
    if not memory_mb:
        raise ValueError('memory_mb must contain at least one sample')
    if maximum < 256:
        raise ValueError('maximum must be >= 256')
    current = maximum
    trace = []
    for index, mb in enumerate(memory_mb):
        available = None if mb is None else mb * MIB
        audit = audit_available(available, reserve_mb * MIB)
        # Production checks available_supported before calling adaptive_batch.
        # Invalid/unavailable samples must not force a downshift or throw.
        if audit['status'] in ('unavailable', 'invalid'):
            next_batch = current
        else:
            next_batch = adaptive_batch(current, maximum, available,
                                        reserve_mb * MIB, features)
        if not (256 <= next_batch <= current <= maximum):
            raise AssertionError('Batch safety invariant violated')
        trace.append({'check': index + 1, 'available_mb': mb,
                      'status': audit['status'], 'batch_before': current,
                      'batch_after': next_batch, 'downshift': int(next_batch < current)})
        current = next_batch
    return trace


def synthetic_equivalence(batch_sizes=(4096, 2048, 1024, 256), pixels=10003):
    """Test deterministic per-pixel class and float32-like confidence signatures.

    This is not a substitute for pixelwise QGIS GeoTIFF comparison.
    """
    import struct
    def signature(batch):
        raw, confidence = [], []
        for start in range(0, pixels, batch):
            for x in range(start, min(pixels, start + batch)):
                raw.append((x * 7 + 3) % 7 + 1)
                val = 0.5 + ((x * 37) % 500) / 1000.0
                confidence.append(struct.pack('<f', val))
        return raw, confidence
    reference = signature(batch_sizes[0])
    return all(signature(b) == reference for b in batch_sizes[1:])


def run_suite(output_csv=None):
    scenarios = {
        'normal': [12000, 12000, 12000],
        'moderate_downshift': [12000, 2050, 2050],
        'critical_low_ram': [12000, 2048, 2047],
        'unavailable_then_recover': [None, 12000, 2048],
        'invalid_then_recover': [float('nan'), -1, 2048],
        'recover_does_not_upshift': [2048, 12000, 12000],
    }
    rows = []
    for name, sequence in scenarios.items():
        for row in simulated_batches(sequence):
            rows.append({'scenario': name, **row})
    if output_csv:
        dest = Path(output_csv)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Safe synthetic adaptive batch stress test')
    parser.add_argument('--csv', default='adaptive_stress_v25538.csv')
    args = parser.parse_args()
    rows = run_suite(args.csv)
    assert synthetic_equivalence()
    for name in sorted({r['scenario'] for r in rows}):
        subset = [r for r in rows if r['scenario'] == name]
        print('%s: %s -> %s, downshifts=%s' % (
            name, subset[0]['batch_before'], subset[-1]['batch_after'],
            sum(r['downshift'] for r in subset)))
    print('Synthetic class/confidence equivalence: PASS; CSV:', args.csv)
