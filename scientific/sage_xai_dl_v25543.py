"""SAGE-GeoClass 2.5.5.4.3 DL XAI: bounded, auditable grouped interventions.
No TensorFlow worker changes. Explanations operate on predict probabilities;
spatial patches are grouped by spectral channel, temporal tensors by time/channel.
"""
import csv
import numpy as np

def _group_axes(x, key):
    if key == 'SpatialCNN':
        if x.ndim != 4: raise ValueError('SpatialCNN expects (N,H,W,C)')
        return x.shape[-1]
    if key == 'TemporalCNN':
        if x.ndim != 3: raise ValueError('TemporalCNN expects (N,T,C)')
        return x.shape[1]*x.shape[2]
    if key == 'DNN':
        if x.ndim != 2: raise ValueError('DNN expects (N,C)')
        return x.shape[1]
    raise ValueError('Unsupported DL key')

def _view(x,key):
    if key=='SpatialCNN':return x.reshape((len(x),-1,x.shape[-1]))
    return x.reshape((len(x),-1))

def _intervene(x,baseline,key,gate):
    x=np.asarray(x,dtype=np.float32)
    y=x.copy()
    if key=='SpatialCNN':
        for j in range(x.shape[-1]):
            if gate[j]==0:y[...,j]=baseline[...,j]
    else:
        a=y.reshape(len(y),-1);b=baseline.reshape(len(baseline),-1)
        for j in range(a.shape[1]):
            if gate[j]==0:a[:,j]=b[:,j]
    return y

def _predict(model,x,key,scaler=None,batch=128):
    results=[]
    for i in range(0,len(x),batch):
        v=x[i:i+batch]
        if key=='DNN' and scaler is not None:v=scaler.transform(v)
        p=np.asarray(model.predict(v,verbose=0))
        if p.ndim!=2 or not np.isfinite(p).all():raise ValueError('Invalid DL probabilities')
        results.append(p)
    return np.concatenate(results)

def _subset(x,cap,seed):
    if len(x)<=cap:return np.asarray(x)
    ids=np.sort(np.random.default_rng(seed).choice(len(x),cap,replace=False))
    return np.asarray(x)[ids]

def _names(names,n,key):
    if key=='SpatialCNN':
        return [names[j] if j<len(names) else 'Band_%d'%(j+1) for j in range(n)]
    if len(names)==n:return list(names)
    return ['TimeBand_%d'%(j+1) for j in range(n)] if key=='TemporalCNN' else ['Feature_%d'%(j+1) for j in range(n)]

def grouped_kernel_shap(model,train,val,names,output,key,scaler=None,seed=42,cap=16,
                        sample_output=None,classwise_output=None,color_mode='PATCH_MEAN'):
    import shap
    train=np.asarray(train,dtype=np.float32);val=np.asarray(val,dtype=np.float32)
    n=_group_axes(val,key)
    if _group_axes(train,key)!=n:raise ValueError('Train/validation group mismatch')
    if n>32:raise ValueError('Too many grouped dimensions for bounded KernelSHAP (>32)')
    fg=_subset(val,cap,seed);bg=_subset(train,64,seed+1)
    # One baseline patch/tensor per foreground sample: TRAIN median template.
    base=np.broadcast_to(np.median(bg,axis=0).astype(np.float32),fg.shape)
    # Evaluate each foreground independently so coalition rows map to the
    # exact same sample; never silently repeat or mix spatial patches.
    vals=[]; signed_rows=[]
    for i in range(len(fg)):
        def predict_gates(gates):
            gates=np.asarray(gates)
            xs=np.repeat(fg[i:i+1],len(gates),axis=0)
            bs=np.repeat(base[i:i+1],len(gates),axis=0)
            out=np.stack([_intervene(xs[j:j+1],bs[j:j+1],key,g) [0] for j,g in enumerate(gates)])
            return _predict(model,out,key,scaler=scaler)
        explainer=shap.KernelExplainer(predict_gates,np.zeros((1,n),dtype=np.float32))
        v=explainer.shap_values(np.ones((1,n),dtype=np.float32),nsamples=max(64,2*n+32),silent=True)
        a=np.asarray(v)
        if isinstance(v,list):a=np.stack(v,axis=-1)
        if a.ndim==2:a=a[:,:,None]
        if a.ndim!=3 or a.shape[0]!=1 or a.shape[1]!=n:
            raise ValueError('Unexpected signed grouped SHAP shape: %s'%(a.shape,))
        signed_rows.append(a[0])
        if a.ndim==3:imp=np.abs(a).mean(axis=(0,2))
        elif a.ndim==2:imp=np.abs(a).mean(axis=0)
        else:raise ValueError('Unexpected SHAP shape %s'%(a.shape,))
        if imp.shape!=(n,):raise ValueError('SHAP group count mismatch')
        vals.append(imp)
    means=np.mean(vals,axis=0)
    if sample_output or classwise_output:
        import importlib.util as _ilu, os as _os
        _vp=_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "sage_shap_visualization_v255432.py")
        _vs=_ilu.spec_from_file_location("sage_shap_visualization_v255432",_vp)
        _vm=_ilu.module_from_spec(_vs);_vs.loader.exec_module(_vm)
        write_samples,write_classwise=_vm.write_samples,_vm.write_classwise
        signed=np.stack(signed_rows,axis=0)
        if key=='SpatialCNN':
            if color_mode=='PATCH_MEAN': feature_values=fg.mean(axis=(1,2))
            elif color_mode=='CENTER_PIXEL':feature_values=fg[:,fg.shape[1]//2,fg.shape[2]//2,:]
            elif color_mode=='PATCH_MEDIAN':feature_values=np.median(fg,axis=(1,2))
            elif color_mode=='PATCH_STD':feature_values=fg.std(axis=(1,2))
            else:raise ValueError('Unsupported SHAP color mode')
        else:
            feature_values=fg.reshape((len(fg),n))
        labels=list(range(1,signed.shape[-1]+1))
        if sample_output:write_samples(sample_output,feature_values,signed,_names(names,n,key),labels,key,color_mode if key=='SpatialCNN' else 'FEATURE_VALUE')
        if classwise_output:write_classwise(classwise_output,signed,_names(names,n,key),labels,key)
    with open(output,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['feature','mean_absolute_shap','method','sample_count','model','output_space','group_definition','reference_partition'])
        for name,v in zip(_names(names,n,key),means):
            w.writerow([name,float(v),'GroupedKernelSHAP',len(fg),key,'class_probability', 'whole_spatial_channel' if key=='SpatialCNN' else 'flattened_time_channel', 'train'])
    return means

def grouped_spatial_importance(model,train,val,y,groups,names,output,key,scaler=None,seed=42,cap=1500,repeats=2):
    from sklearn.metrics import accuracy_score
    val=np.asarray(val,dtype=np.float32);y=np.asarray(y);groups=np.asarray(groups)
    n=_group_axes(val,key)
    if len(val)!=len(y) or len(val)!=len(groups):raise ValueError('Validation/group shape mismatch')
    rng=np.random.default_rng(seed);ids=np.arange(len(val))
    if len(ids)>cap:ids=np.sort(rng.choice(ids,cap,replace=False))
    x=val[ids];y=y[ids];groups=groups[ids]
    pred=np.argmax(_predict(model,x,key,scaler=scaler),axis=1)+1
    # Class IDs in this pipeline are contiguous 1..K.
    base=accuracy_score(y,pred);rows=[]
    for j,name in enumerate(_names(names,n,key)):
        drops=[];weights=[]
        for group in np.unique(groups):
            ix=np.flatnonzero(groups==group)
            if len(ix)<2:continue
            local=x[ix];target=y[ix]
            baseline=accuracy_score(target,np.argmax(_predict(model,local,key,scaler=scaler),axis=1)+1)
            for _ in range(repeats):
                changed=local.copy();order=rng.permutation(len(ix))
                if key=='SpatialCNN':changed[...,j]=local[order,...,j]
                else:
                    a=changed.reshape(len(ix),-1);b=local.reshape(len(ix),-1)
                    a[:,j]=b[order,j]
                acc=accuracy_score(target,np.argmax(_predict(model,changed,key,scaler=scaler),axis=1)+1)
                drops.append(baseline-acc);weights.append(len(ix))
        rows.append([name,float(base),float(np.average(drops,weights=weights)) if drops else float('nan'),len(np.unique(groups)),len(drops),len(x)])
    with open(output,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f);w.writerow(['feature','baseline_accuracy','mean_group_permutation_drop','groups_total','permutations_evaluated','sample_count']);w.writerows(rows)
    return rows
