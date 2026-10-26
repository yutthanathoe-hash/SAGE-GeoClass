import hashlib
import json
from pathlib import Path

def verify():
    root = Path(__file__).resolve().parent
    expected = json.loads((root / 'SCIENTIFIC_SHA256.json').read_text(encoding='utf-8'))['scientific_files_sha256']
    return {name: hashlib.sha256((root / 'scientific' / name).read_bytes()).hexdigest() == digest for name, digest in expected.items()}

if __name__ == '__main__':
    results = verify()
    for name, ok in sorted(results.items()):
        print(('PASS' if ok else 'FAIL'), name)
    raise SystemExit(0 if all(results.values()) else 1)
