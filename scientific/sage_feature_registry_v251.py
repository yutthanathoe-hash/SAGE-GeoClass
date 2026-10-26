"""SAGE-GeoClass V2.5.1 feature schema. Pure Python; no QGIS/TensorFlow imports."""
import json
import re
import hashlib

ROLES = {'blue','green','red','nir','swir1','swir2','vv','vh','ndvi','ndwi','elevation','slope','aspect','texture','categorical','continuous','other'}
KINDS = {'continuous','categorical'}

def build_registry(sources, schema_text=''):
    """sources: [{path, bands:[{nodata,description}]}] in model feature order."""
    raw = json.loads(schema_text) if schema_text and schema_text.strip() else {}
    if not isinstance(raw,dict): raise ValueError('Feature schema must be a JSON object')
    overrides = raw.get('features',[])
    if not isinstance(overrides,list): raise ValueError('features must be a list')
    entries=[]
    for source_index,source in enumerate(sources,1):
        for band_index,band in enumerate(source['bands'],1):
            entries.append({'index':len(entries)+1,'source_index':source_index,'source_path':source['path'],
                'band':band_index,'name':f'Raster{source_index}_Band{band_index}',
                'role':'other','kind':'continuous','unit':'','nodata':band.get('nodata'),
                'description':band.get('description','')})
    if overrides and len(overrides)!=len(entries):
        raise ValueError(f'Feature schema has {len(overrides)} entries but raster stack has {len(entries)} bands')
    allowed={'index','name','role','kind','unit'}
    for i,override in enumerate(overrides):
        if not isinstance(override,dict): raise ValueError('Each feature entry must be an object')
        if set(override)-allowed: raise ValueError(f'Unknown feature keys: {sorted(set(override)-allowed)}')
        if 'index' in override and override['index']!=i+1: raise ValueError('Feature index must match raster stacking order')
        for key in ('name','role','kind','unit'):
            if key in override: entries[i][key]=override[key]
    names=[]
    for entry in entries:
        if not isinstance(entry['name'],str) or not entry['name'].strip(): raise ValueError('Empty feature name')
        if entry['role'] not in ROLES: raise ValueError(f"Unknown role: {entry['role']}")
        if entry['kind'] not in KINDS: raise ValueError(f"Unknown kind: {entry['kind']}")
        if not isinstance(entry['unit'],str): raise ValueError('Unit must be a string')
        names.append(entry['name'].casefold())
    if len(names)!=len(set(names)): raise ValueError('Duplicate feature names')
    roles={}
    for e in entries:
        if e['role'] in {'blue','green','red','nir'}:
            if e['role'] in roles: raise ValueError(f"Duplicate spectral role: {e['role']}")
            roles[e['role']]=e['index']
    canonical=json.dumps(entries,sort_keys=True,ensure_ascii=False,default=str)
    return {'version':'2.5.1','features':entries,'spectral_indices':[roles[k] for k in ('blue','green','red','nir')] if len(roles)==4 else None,
            'schema_sha256':hashlib.sha256(canonical.encode('utf-8')).hexdigest()}
