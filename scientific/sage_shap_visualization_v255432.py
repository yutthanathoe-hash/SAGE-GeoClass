"""SAGE-GeoClass 2.5.5.4.3.2 optional signed SHAP export and plots.
Read-only post-training diagnostics. No mutation of model or raster arrays.
"""
import csv
import os
import numpy as np

def write_samples(path, values, shap_values, feature_names, class_ids, model_name, value_mode):
    v=np.asarray(values);s=np.asarray(shap_values)
    if s.ndim!=3 or v.shape!=s.shape[:2] or s.shape[2]!=len(class_ids):
        raise ValueError('SHAP sample feature/class shape mismatch')
    if not np.isfinite(s).all():raise ValueError('Nonfinite SHAP values')
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f)
        w.writerow(['sample_id','class_id','feature','shap_value','feature_value','feature_value_mode','model','output_space'])
        for i in range(len(v)):
            for k,c in enumerate(class_ids):
                for j,name in enumerate(feature_names):
                    w.writerow([i,c,name,float(s[i,j,k]),float(v[i,j]),value_mode,model_name,
                                'class_probability' if model_name in ('SVM','MLP','SpatialCNN','TemporalCNN','DNN') else 'model_native'])

def write_classwise(path, shap_values, feature_names, class_ids, model_name):
    s=np.asarray(shap_values)
    with open(path,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.writer(f)
        w.writerow(['class_id','feature','mean_absolute_shap','mean_signed_shap','sample_count','model'])
        for k,c in enumerate(class_ids):
            for j,name in enumerate(feature_names):
                w.writerow([c,name,float(np.mean(np.abs(s[:,j,k]))),float(np.mean(s[:,j,k])),len(s),model_name])

def make_plots(samples_csv, summary_csv, beeswarm_png=None, bar_png=None, model_name='',color_mode='PATCH_MEAN',target_class=''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    from collections import defaultdict
    rows=[]
    if beeswarm_png:
        if not samples_csv or not os.path.isfile(samples_csv):
            raise ValueError('Beeswarm needs OUTPUT_SHAP_SAMPLES (signed sample-level values)')
        with open(samples_csv,encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        if not rows:raise ValueError('Empty SHAP sample CSV')
        # Plot a SINGLE class to avoid mixing opposite signed class contributions.
        byclass=defaultdict(list)
        for r in rows:byclass[r['class_id']].append(r)
        target=max(sorted(byclass),key=lambda k:sum(abs(float(r['shap_value'])) for r in byclass[k]))
        if target_class:
            if str(target_class) not in byclass:
                raise ValueError('Requested SHAP target class %s not in sample data' % target_class)
            target=str(target_class)
        data=byclass[target];features=list(dict.fromkeys(r['feature'] for r in data))
        groups={name:[r for r in data if r['feature']==name] for name in features}
        features=sorted(features,key=lambda name:np.mean([abs(float(r['shap_value'])) for r in groups[name]]),reverse=True)
        fig,ax=plt.subplots(figsize=(10,max(4.2,.5*len(features)+1.7)))
        cmap=plt.get_cmap('coolwarm');rng=np.random.default_rng(42)
        for j,name in enumerate(features):
            rr=groups[name];xx=np.array([float(r['shap_value']) for r in rr]);vv=np.array([float(r['feature_value']) for r in rr])
            lo,hi=np.nanpercentile(vv,[5,95]);norm=np.clip((vv-lo)/(hi-lo),0,1) if hi>lo else np.full(len(vv),.5)
            ax.scatter(xx,j+rng.uniform(-.23,.23,len(xx)),c=cmap(norm),s=13,alpha=.82,linewidths=0,rasterized=True)
        ax.axvline(0,color='gray',lw=.9);ax.set_yticks(range(len(features)),features);ax.invert_yaxis()
        ax.set_xlabel('Signed SHAP value (model output impact)')
        ax.set_title(f'{model_name} SHAP Beeswarm — Class {target}\nColor: {color_mode}; N={len(groups[features[0]])}')
        ax.grid(axis='y',alpha=.2)
        sm=plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0,vmax=1),cmap=cmap)
        cb=fig.colorbar(sm,ax=ax,pad=.02);cb.set_label('Feature value (within-feature normalized)');cb.set_ticks([0,1],labels=['Low','High'])
        fig.tight_layout();fig.savefig(beeswarm_png,dpi=300,bbox_inches='tight');plt.close(fig)
    if bar_png:
        with open(summary_csv,encoding='utf-8-sig',newline='') as f:rr=list(csv.DictReader(f))
        rr=[r for r in rr if 'feature' in r and 'mean_absolute_shap' in r]
        rr=sorted(rr,key=lambda r:float(r['mean_absolute_shap']),reverse=True)
        if not rr:raise ValueError('Empty SHAP summary')
        fig,ax=plt.subplots(figsize=(9,max(3.5,.45*len(rr)+1.4)))
        ax.barh([r['feature'] for r in rr],[float(r['mean_absolute_shap']) for r in rr])
        ax.invert_yaxis();ax.set_xlabel('Mean absolute SHAP');ax.set_title(f'{model_name} — Global SHAP importance')
        fig.tight_layout();fig.savefig(bar_png,dpi=300,bbox_inches='tight');plt.close(fig)
