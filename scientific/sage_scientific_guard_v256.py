"""Additive scientific guard; never modifies frozen core or its worker."""
import csv, hashlib, importlib.util, json, os, sys, time
from pathlib import Path

MODEL_REQUIREMENTS = {'Random Forest':('sklearn',), 'Extra Trees':('sklearn',), 'HistGradientBoosting':('sklearn',), 'SVM (RBF)':('sklearn',), 'MLP Neural Network':('sklearn',), 'Spatial 2D-CNN (TensorFlow)':('tensorflow',), 'Temporal CNN (TensorFlow)':('tensorflow',), 'Deep Neural Network (TensorFlow DNN)':('tensorflow',), 'XGBoost':('xgboost',), 'LightGBM':('lightgbm',), 'CatBoost':('catboost',), 'Logistic Regression':('sklearn',)}

def dependency_matrix(temporal_steps=1, spatial_enabled=False, temporal_enabled=False):
    rows=[]
    for name, deps in MODEL_REQUIREMENTS.items():
        missing=[dep for dep in deps if importlib.util.find_spec(dep) is None]
        reason='missing packages: '+', '.join(missing) if missing else ''
        status='SKIPPED_DEPENDENCY' if missing else 'READY'
        if not missing and name.startswith('Temporal CNN') and (not temporal_enabled or temporal_steps<2):
            status='INVALID_INPUT';reason='Temporal CNN requires enabled temporal input and >=2 dates'
        if not missing and name.startswith('Spatial 2D-CNN') and not spatial_enabled:
            status='INVALID_INPUT';reason='Spatial CNN disabled'
        rows.append({'model':name,'status':status,'reason':reason,'dependencies':','.join(deps),'python':sys.version.split()[0]})
    return rows

def write_csv(path,rows,fields):
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def audit_outputs(directory, strict=False):
    root=Path(directory); checks=[]
    def add(check,status,detail):checks.append({'check':check,'status':status,'detail':str(detail)})
    def load(name):
        p=root/name
        if not p.is_file():add(name,'MISSING','Output file not present');return []
        try:
            with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
        except Exception as exc:add(name,'FAIL',repr(exc));return []
    comp=load('OUTPUT_MODEL_COMPARISON.csv')
    if comp:
        ok=[r for r in comp if r.get('status')=='OK']
        add('comparison_status','PASS' if ok else 'FAIL',f'{len(ok)} successful of {len(comp)} attempted; no failed model is scored as zero')
        add('all_12_models_evaluated','PASS' if len(ok)==12 else 'INCOMPLETE',f'{len(ok)}/12 succeeded; auto compare is not a 12-model controlled spatial-CV benchmark')
        for r in comp:
            if r.get('status')!='OK':add('model_'+r.get('model','unknown'),'INCOMPLETE',r.get('status','unknown'))
    cv=load('OUTPUT_CV.csv')
    folds=[r for r in cv if str(r.get('fold','')).isdigit()]
    if folds:
        bad=[r for r in folds if r.get('status')!='PASS' or any(float(r.get(k) or 0)!=0 for k in ('patch_overlap_after','group_overlap_after'))]
        add('spatial_cv_leakage','FAIL' if bad else 'PASS',f'{len(folds)} folds; {len(bad)} failures/overlaps')
        add('spatial_cv_12_models','INCOMPLETE' if len(set(r.get('model') for r in folds))<12 else 'PASS',f'{len(set(r.get("model") for r in folds))} model(s) have fold records')
    qa=load('OUTPUT_RASTER_QA.csv')
    if qa:add('raster_qa_available','PASS',f'{len(qa)} QA records; review assertions and masks')
    else:add('raster_qa_available','INCOMPLETE','No readable QA records')
    split=load('OUTPUT_SPLIT.csv')
    add('split_audit_present','PASS' if split else 'INCOMPLETE',f'{len(split)} rows')
    xai=load('OUTPUT_XAI_AUDIT.csv')
    add('xai_audit_present','PASS' if xai else 'INCOMPLETE',f'{len(xai)} rows; numeric additivity not assumed from file presence')
    spectral=load('OUTPUT_SPECTRAL_QA.csv')
    add('spectral_roles_verified','INCOMPLETE','Explicit B,G,R,NIR mapping must be confirmed against input raster metadata; no automatic PASS')
    add('independent_test_isolation','NOT_VERIFIED','Requires audit of train/validation/test indices and selection provenance; logs alone insufficient')
    add('ood_trust_interpretation','INFO','OOD=feature distance; TRUST=heuristic score, not calibrated probabilities')
    summary={'schema':'SAGE-GeoClass-V2.5.5.4.3.6-scientific-audit','timestamp_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'output_directory':str(root),'checks':checks,'counts':{s:sum(r['status']==s for r in checks) for s in set(r['status'] for r in checks)}}
    (root/'OUTPUT_SCIENTIFIC_AUDIT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    write_csv(root/'OUTPUT_SCIENTIFIC_AUDIT.csv',checks,['check','status','detail'])
    if strict and any(c['status'] in ('FAIL','MISSING','INCOMPLETE','NOT_VERIFIED') for c in checks):raise RuntimeError('Scientific audit not fully passed; inspect OUTPUT_SCIENTIFIC_AUDIT.csv')
    return summary

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
