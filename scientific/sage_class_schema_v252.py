"""SAGE-GeoClass V2.5.2 universal class schema (standard-library only)."""
import hashlib
import json
import re

_HEX = re.compile(r'^#[0-9a-fA-F]{6}$')
_NULL = {'', 'NULL', 'NONE', 'NAN', '<NULL>'}

def normalize_label(value):
    if value is None:
        return None
    label = str(value).strip()
    return None if label.upper() in _NULL else label

def build_class_schema(values, schema_text='', min_samples=20):
    labels = {normalize_label(v) for v in values}
    labels.discard(None)
    if len(labels) < 2:
        raise ValueError('At least two non-null training classes are required')
    def sort_key(s):
        return (0, int(s)) if re.fullmatch(r'[+-]?\d+', s) else (1, s.casefold(), s)
    ordered = sorted(labels, key=sort_key)
    raw = json.loads(schema_text) if schema_text and schema_text.strip() else {}
    if not isinstance(raw, dict):
        raise ValueError('CLASS_SCHEMA_JSON must be a JSON object')
    if set(raw) - {'classes'}:
        raise ValueError('Unknown top-level schema keys: ' + repr(sorted(set(raw)-{'classes'})))
    supplied = raw.get('classes', [])
    if not isinstance(supplied, list):
        raise ValueError('classes must be an array')
    config = {}
    for entry in supplied:
        if not isinstance(entry, dict) or set(entry) - {'value', 'name', 'color', 'description'}:
            raise ValueError('Each class must contain only value/name/color/description')
        label = normalize_label(entry.get('value'))
        if label is None or label in config:
            raise ValueError('Missing or duplicated original class value')
        if label not in labels:
            raise ValueError('Schema class not found in training data: ' + label)
        config[label] = entry
    if supplied and set(config) != labels:
        raise ValueError('Schema must include every observed class. Missing: ' + repr(sorted(labels-set(config))))
    entries = []
    for i, label in enumerate(ordered, 1):
        entry = config.get(label, {})
        name = entry.get('name', label)
        color = entry.get('color', '#808080')
        desc = entry.get('description', '')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Class name must be nonempty: ' + label)
        if not isinstance(color, str) or not _HEX.fullmatch(color):
            raise ValueError('Invalid #RRGGBB color: ' + label)
        if not isinstance(desc, str):
            raise ValueError('Description must be text: ' + label)
        entries.append({'dn':i, 'value':label, 'name':name.strip(), 'color':color.upper(), 'description':desc})
    names = [e['name'].casefold() for e in entries]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate display names are not permitted')
    digest = hashlib.sha256(json.dumps(entries, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
    return {'version':'2.5.2', 'classes':entries, 'class_count':len(entries), 'schema_sha256':digest,
            'minimum_sample_pixels_for_qc':int(min_samples)}

def class_qc(schema, y, groups, minimum=20):
    """Compute descriptive QC; never silently drop a class."""
    import numpy as np
    results=[]
    for e in schema['classes']:
        mask = y == e['dn']
        n = int(np.count_nonzero(mask))
        g = int(np.unique(groups[mask]).size)
        warnings=[]
        if n < minimum: warnings.append('LOW_SAMPLE_PIXELS')
        if g < 3: warnings.append('FEWER_THAN_3_GROUPS')
        results.append({'dn':e['dn'], 'value':e['value'], 'name':e['name'],
                        'sample_pixels':n, 'independent_groups':g,
                        'status':'WARN' if warnings else 'PASS', 'warnings':';'.join(warnings)})
    return results
