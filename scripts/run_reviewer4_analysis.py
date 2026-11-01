"""Reviewer 4 analyses. Reads prior artifacts; writes only the new revision directory."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import argparse,json,sys,hashlib,itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'application'))
from src.utils.config import FEATURE_COLS_V3
from src.evaluation.protocol import grouped_train_val_test_folds
from scripts.run_protocol_comparison import _add_legacy_target_proxies,_chronological_indices,LEAKAGE_PROXY_COLS
BASE=ROOT/'application/artifacts/v3';RESULTS=BASE/'results'
REV=ROOT/'paper/revision-2026-11-r4';OUT=REV/'reproducibility';DATA=OUT/'tables';PRED=OUT/'predictions'
NAMES=dict(pd.read_csv(RESULTS/'model_catalog.csv')[['model_id','display_name']].values) if (RESULTS/'model_catalog.csv').exists() else {}

def add_proxies(d):return _add_legacy_target_proxies(d)
def write(d,name):
    DATA.mkdir(parents=True,exist_ok=True);d.to_csv(DATA/(name+'.csv'),index=False)
def model(seed):return Pipeline([('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),('model',ExtraTreesRegressor(n_estimators=300,min_samples_leaf=2,random_state=seed,n_jobs=4))])
def aggregate_errors(d):
    g=d.groupby(['battery_id','cycle_number'],as_index=False).agg(y_true=('y_true','first'),y_pred=('y_pred','mean'))
    g['ae']=abs(g.y_true-g.y_pred)
    return {'sample_mae':float(g.ae.mean()),'battery_macro_mae':float(g.groupby('battery_id').ae.mean().mean()),'n_cycles':len(g),'n_batteries':g.battery_id.nunique()}
def bootstrap_effect(d,n=10000):
    grouped=d.groupby('battery_id').agg(base=('base_abs_error','sum'),altered=('altered_abs_error','sum'),count=('base_abs_error','size'))
    delta=(grouped.altered-grouped.base).to_numpy();count=grouped['count'].to_numpy();rng=np.random.default_rng(42)
    ix=rng.integers(0,len(grouped),size=(n,len(grouped)))
    draws=delta[ix].sum(axis=1)/count[ix].sum(axis=1)
    return {'estimate':float(delta.sum()/count.sum()),'ci_low':float(np.quantile(draws,.025)),'ci_high':float(np.quantile(draws,.975)),'n_batteries':len(grouped),'n_cycles':len(d)}
def permuted_proxies(x,partitions,seed):
    out=x.copy();rng=np.random.default_rng(seed)
    for ix in partitions:
        for j in range(x.shape[1]):out[ix,j]=x[rng.permutation(ix),j]
    return out
def health(values):return np.where(values>=90,0,np.where(values>=80,1,np.where(values>=70,2,3)))

def metadata():
    f=pd.read_csv(BASE/'features/nasa/features.csv');groups=f.battery_id.to_numpy();rows=[];members=[]
    for seed in [17,42,2026]:
        for fold,(tr,va,te) in enumerate(grouped_train_val_test_folds(groups,n_splits=5,random_state=seed),1):
            assert not(set(groups[tr])&set(groups[va]) or set(groups[tr])&set(groups[te]) or set(groups[va])&set(groups[te]))
            cached=json.loads((ROOT/f'paper/revision-2026-09/restoration/evaluation/split_s{seed}_f{fold}.json').read_text())
            for role,ix in [('train',tr),('validation',va),('test',te)]:
                ids=sorted(set(groups[ix]));assert ids==cached[role]
                d=f.iloc[ix];cls=health(d.SoH)
                record={'seed':seed,'fold':fold,'role':role,'n_batteries':len(ids),'n_cycles':len(ix),'battery_ids':'; '.join(ids),'soh_min':d.SoH.min(),'soh_q25':d.SoH.quantile(.25),'soh_median':d.SoH.median(),'soh_q75':d.SoH.quantile(.75),'soh_max':d.SoH.max(),'temperature_values':'; '.join(map(str,sorted(d.ambient_temperature.unique()))),'current_c_rate_min':d.current_mean_c_rate.min(),'current_c_rate_max':d.current_mean_c_rate.max()}
                record.update({name:int((cls==i).sum()) for i,name in enumerate(['healthy','aging','near_eol','eol'])});rows.append(record)
                for ix0 in ix:members.append({'seed':seed,'fold':fold,'role':role,'row_index':int(ix0),'battery_id':groups[ix0],'cycle_number':int(f.iloc[ix0].cycle_number),'soh':float(f.iloc[ix0].SoH),'ambient_temperature':float(f.iloc[ix0].ambient_temperature),'class_id':int(health(f.iloc[ix0].SoH))})
    write(pd.DataFrame(rows),'nasa_split_inventory');write(pd.DataFrame(members),'nasa_split_records')
    epoch=[]
    for dataset in ['nasa','calce','oxford']:
        for family in ['recurrent','transformer','graph_variational']:
            d=pd.read_csv(RESULTS/f'{dataset}_{family}_fold_metrics.csv');epoch.append(d.assign(artifact_family=family,provenance='original_regression_benchmark'))
    epoch=pd.concat(epoch,ignore_index=True);write(epoch,'original_deep_fold_stopping')
    summary=epoch.groupby(['dataset','model']).agg(fits=('epochs','size'),epochs_min=('epochs','min'),epochs_median=('epochs','median'),epochs_max=('epochs','max'),epoch_cap_hits=('epochs',lambda v:int((v==200).sum())),train_mae=('train_mae','mean'),test_mae=('mae','mean'),test_minus_train_mae=('generalization_gap_mae','mean')).reset_index();write(summary,'original_deep_stopping_summary')
    gaps=[]
    for path in sorted((ROOT/'paper/revision-2026-09/restoration/evaluation').glob('*.json')):
        m=json.loads(path.read_text())
        if 'epochs' not in m:continue
        d=pd.read_csv(path.with_suffix('.csv'),float_precision='round_trip');v=d[d.role=='validation'];t=d[d.role=='test']
        vm=abs(v.y_true-v.y_pred).mean();tm=abs(t.y_true-t.y_pred).mean()
        gaps.append({'model':m['model'],'seed':m['seed'],'fold':m['fold'],'epochs':m['epochs'],'validation_mae':vm,'test_mae':tm,'test_minus_validation_mae':tm-vm,'validation_batteries':v.battery_id.nunique(),'test_batteries':t.battery_id.nunique(),'provenance':'secondary_NASA_rerun'})
    gaps=pd.DataFrame(gaps);write(gaps,'secondary_nasa_validation_test_gaps')
    gs=gaps.groupby('model').agg(fits=('epochs','size'),epochs_min=('epochs','min'),epochs_median=('epochs','median'),epochs_max=('epochs','max'),validation_mae=('validation_mae','mean'),test_mae=('test_mae','mean'),test_minus_validation_mae=('test_minus_validation_mae','mean')).reset_index();write(gs,'secondary_nasa_gap_summary')
    ensemble=[]
    for seed in [17,42,2026]:
        for fold,(tr,va,te) in enumerate(grouped_train_val_test_folds(groups,n_splits=5,random_state=seed),1):
            ids=['extra_trees','gradient_boosting','random_forest','xgboost','lightgbm','svr','ridge','knn']
            predictions=[]
            for name in ids:
                d=pd.read_csv(ROOT/f'paper/revision-2026-09/restoration/evaluation/{name}_s{seed}_f{fold}.csv',float_precision='round_trip');predictions.append(d[d.role=='validation'].y_pred.to_numpy())
            V=np.column_stack(predictions);err=np.maximum(np.mean(abs(V-f.iloc[va].SoH.to_numpy()[:,None]),axis=0),1e-8);w=1/err;w/=w.sum()
            for name,weight in zip(ids,w):ensemble.append({'seed':seed,'fold':fold,'base_model':name,'weight':weight,'fit_partition':'validation','test_batteries_accessed':False})
    write(pd.DataFrame(ensemble),'secondary_ensemble_fold_weights')
    # Leave-one-cell-out sensitivity conditional on already-fitted grouped predictions.
    avg=pd.read_csv(RESULTS/'averaged_predictions.csv');cal=avg[(avg.dataset.str.lower()=='calce')&(avg.model=='stacking_ensemble')].copy()
    fold_metrics=pd.concat([pd.read_csv(RESULTS/f'nasa_{family}_fold_metrics.csv') for family in ['classical','recurrent','transformer','graph_variational']],ignore_index=True)
    metrics=[]
    for name,d in avg[avg.dataset.str.lower()=='nasa'].groupby('model'):
        metrics.append({'model':name,**aggregate_errors(d),'fold_mean_mae':fold_metrics[fold_metrics.model==name].mae.mean()})
    write(pd.DataFrame(metrics),'nasa_metric_estimands')
    loo=[]
    for cell in sorted(cal.battery_id.unique()):
        d=cal[cal.battery_id!=cell];loo.append({'omitted_battery':cell,**aggregate_errors(d)})
    write(pd.DataFrame(loo),'calce_leave_one_battery_out_sensitivity')
    print('METADATA',len(rows),'splits/roles;',len(epoch),'original deep fits;',len(gaps),'secondary fits',flush=True)

def controls(temperature_match=False):
    PRED.mkdir(parents=True,exist_ok=True)
    f=add_proxies(pd.read_csv(BASE/'features/nasa/features.csv'));g=f.battery_id.to_numpy();y=f.SoH.to_numpy();splits=[]
    for seed in [17,42,2026]:
        tr,te=train_test_split(np.arange(len(f)),test_size=.2,random_state=seed);splits.append((seed,'v1_random_cycle',1,tr,np.array([],int),te))
        tr,te=_chronological_indices(f);splits.append((seed,'v2_chronological',1,tr,np.array([],int),te))
        for fold,(tr,va,te) in enumerate(grouped_train_val_test_folds(g,n_splits=5,random_state=seed),1):splits.append((seed,'v3_grouped_battery',fold,tr,va,te))
    prefix='temperature_' if temperature_match else ''
    quotas={float(t):min(int((f.iloc[s[3]].ambient_temperature==t).sum()) for s in splits) for t in sorted(f.ambient_temperature.unique())}
    common=sum(quotas.values()) if temperature_match else min(len(s[3]) for s in splits);records=[];summaries=[]
    for seed,protocol,fold,tr,va,te in splits:
        rng=np.random.default_rng(seed+fold*101)
        selected=np.sort(np.concatenate([rng.choice(tr[f.iloc[tr].ambient_temperature.to_numpy()==t],size=q,replace=False) for t,q in quotas.items() if q])) if temperature_match else np.sort(rng.choice(tr,size=common,replace=False))
        split_id=f'{prefix}{protocol}_s{seed}_f{fold}';split_path=OUT/'controls'/f'{split_id}.json';split_path.parent.mkdir(parents=True,exist_ok=True)
        split_path.write_text(json.dumps({'split_id':split_id,'seed':seed,'fold':fold,'protocol':protocol,'common_train_n':common,'eligible_train_indices':tr.tolist(),'train_indices':selected.tolist(),'validation_indices':va.tolist(),'test_indices':te.tolist(),'train_batteries':sorted(set(g[selected])),'validation_batteries':sorted(set(g[va])),'test_batteries':sorted(set(g[te]))},indent=2))
        safe=f[FEATURE_COLS_V3].to_numpy();proxy=f[LEAKAGE_PROXY_COLS].to_numpy();sham=permuted_proxies(proxy,[selected,te],seed+fold*101)
        for name,x in [('safe18',safe),('safe18_plus4_sham',np.column_stack([safe,sham])),('safe18_plus4_proxies',np.column_stack([safe,proxy]))]:
            path=PRED/f'control_{split_id}_{name}.csv'
            if path.exists():d=pd.read_csv(path,float_precision='round_trip')
            else:
                est=model(seed).fit(x[selected],y[selected]);pred=est.predict(x[te])
                d=f.iloc[te][['battery_id','cycle_number','ambient_temperature','current_mean_c_rate']].copy();d['row_index']=te;d['y_true']=y[te];d['y_pred']=pred;d['split_id']=split_id;d['seed']=seed;d['fold']=fold;d['protocol']=protocol;d['contract']=name;d.to_csv(path,index=False)
            summaries.append({'split_id':split_id,'seed':seed,'fold':fold,'protocol':protocol,'contract':name,'train_n':common,'test_n':len(te),'n_features':x.shape[1],'fold_mae':abs(d.y_true-d.y_pred).mean()})
            records.append(d)
        print('CONTROL',split_id,flush=True)
    allp=pd.concat(records,ignore_index=True);allp.to_csv(PRED/(prefix+'matched_control_predictions.csv'),index=False)
    summary=pd.DataFrame(summaries);write(summary,prefix+'matched_control_fold_metrics');pooled=[];deltas=[];strata=[]
    for (protocol,contract),d in allp.groupby(['protocol','contract']):
        avg=d.groupby(['battery_id','cycle_number','ambient_temperature','current_mean_c_rate'],as_index=False).agg(y_true=('y_true','first'),y_pred=('y_pred','mean'));avg['ae']=abs(avg.y_true-avg.y_pred)
        pooled.append({'protocol':protocol,'contract':contract,'fold_mean_mae':summary[(summary.protocol==protocol)&(summary.contract==contract)].fold_mae.mean(),**aggregate_errors(avg)})
        for category,col in [('temperature','ambient_temperature'),('c_rate','current_mean_c_rate')]:
            labels=avg[col].map(lambda x:f'{x:g} °C') if category=='temperature' else pd.cut(avg[col],[-np.inf,.75,1.25,np.inf],labels=['<0.75 C','0.75–<1.25 C','≥1.25 C'],right=False).astype(str)
            for label,ix in labels.groupby(labels).groups.items():
                dd=avg.loc[ix];strata.append({'protocol':protocol,'contract':contract,'stratum_type':category,'stratum':label,'n_batteries':dd.battery_id.nunique(),'n_cycles':len(dd),'sample_mae':dd.ae.mean()})
    for protocol in allp.protocol.unique():
        pivot=allp[allp.protocol==protocol].groupby(['battery_id','cycle_number','contract']).agg(y_true=('y_true','first'),y_pred=('y_pred','mean')).reset_index();pivot['ae']=abs(pivot.y_true-pivot.y_pred)
        wide=pivot.pivot(index=['battery_id','cycle_number'],columns='contract',values='ae').reset_index()
        for contract in ['safe18_plus4_sham','safe18_plus4_proxies']:
            d=wide.rename(columns={'safe18':'base_abs_error',contract:'altered_abs_error'});deltas.append({'protocol':protocol,'contrast':contract+' minus safe18',**bootstrap_effect(d)})
    write(pd.DataFrame(pooled),prefix+'matched_control_summary');write(pd.DataFrame(deltas),prefix+'matched_control_paired_intervals');write(pd.DataFrame(strata),prefix+'matched_control_temperature_current_strata')
    (OUT/'controls'/f'{prefix}design.json').write_text(json.dumps({'common_train_n':common,'fit_count':len(summary),'n_trees':300,'min_samples_leaf':2,'temperature_quotas':quotas if temperature_match else None,'same_test_rows_within_contrast':True,'matching':'common per-temperature minimum train counts' if temperature_match else 'uniform without-replacement training rows; identical sampled training/test rows for all three contracts within each split','limits':'test populations differ across protocols; no causal attribution of between-protocol MAE. Temperature/current test strata are descriptive; sparse cells are retained. Zero-quota temperatures are excluded from training only in the temperature-matched sensitivity, never from test scoring.'},indent=2))

def importance():
    PRED.mkdir(parents=True,exist_ok=True)
    f=pd.read_csv(BASE/'features/nasa/features.csv');X=f[FEATURE_COLS_V3].to_numpy();y=f.SoH.to_numpy();g=f.battery_id.to_numpy()
    temp=[i for i,c in enumerate(FEATURE_COLS_V3) if 'temperature' in c];voltage=[i for i,c in enumerate(FEATURE_COLS_V3) if 'voltage' in c]
    targets={'temperature_joint':temp,'voltage_joint':voltage,'temperature_voltage_joint':temp+voltage,**{c:[i] for i,c in enumerate(FEATURE_COLS_V3)}}
    effects=[];ablations=[]
    for fold,(tr,_,te) in enumerate(grouped_train_val_test_folds(g,n_splits=5,random_state=42),1):
        path=PRED/f'permutation_f{fold}.csv';abpath=PRED/f'ablation_f{fold}.csv'
        if path.exists() and abpath.exists():effects.append(pd.read_csv(path));ablations.append(pd.read_csv(abpath));continue
        est=model(42).fit(X[tr],y[tr]);base=est.predict(X[te]);babs=abs(y[te]-base);rows=[]
        for name,cols in targets.items():
            for scheme in ['pooled','within_battery']:
                accum=np.zeros(len(te))
                for repeat in range(10):
                    rng=np.random.default_rng(42000+fold*100+repeat);pert=X[te].copy()
                    parts=[np.arange(len(te))] if scheme=='pooled' else [np.flatnonzero(g[te]==cell) for cell in np.unique(g[te])]
                    for ix in parts:
                        donor=rng.permutation(ix);pert[np.ix_(ix,cols)]=X[te][np.ix_(donor,cols)]
                    accum+=abs(y[te]-est.predict(pert))/10
                d=f.iloc[te][['battery_id','cycle_number']].copy();d['row_index']=te;d['fold']=fold;d['feature_or_group']=name;d['scheme']=scheme;d['y_true']=y[te];d['base_abs_error']=babs;d['altered_abs_error']=accum;rows.append(d)
        effects0=pd.concat(rows,ignore_index=True);effects0.to_csv(path,index=False);effects.append(effects0)
        reduced=model(42).fit(X[tr][:,[i for i in range(18) if i not in temp]],y[tr]);pred=reduced.predict(X[te][:,[i for i in range(18) if i not in temp]])
        d=f.iloc[te][['battery_id','cycle_number']].copy();d['row_index']=te;d['fold']=fold;d['y_true']=y[te];d['base_prediction']=base;d['reduced_prediction']=pred;d['base_abs_error']=babs;d['altered_abs_error']=abs(y[te]-pred);d.to_csv(abpath,index=False);ablations.append(d)
        print('IMPORTANCE',fold,flush=True)
    allp=pd.concat(effects,ignore_index=True);allp.to_csv(PRED/'permutation_effect_records.csv',index=False)
    stats=[]
    for (name,scheme),d in allp.groupby(['feature_or_group','scheme']):stats.append({'feature_or_group':name,'scheme':scheme,**bootstrap_effect(d)})
    for scheme in ['pooled','within_battery']:
        d=allp[allp.scheme==scheme];p=d.pivot(index=['battery_id','cycle_number','row_index','fold'],columns='feature_or_group',values='altered_abs_error');base=d.groupby(['battery_id','cycle_number','row_index','fold']).base_abs_error.first();interaction=p.temperature_voltage_joint-p.temperature_joint-p.voltage_joint+base
        z=interaction.rename('altered_abs_error').reset_index();z['base_abs_error']=0;stats.append({'feature_or_group':'interaction_TV_minus_T_minus_V','scheme':scheme,**bootstrap_effect(z)})
    ab=pd.concat(ablations,ignore_index=True);ab.to_csv(PRED/'temperature_ablation_predictions.csv',index=False)
    write(pd.DataFrame(stats),'permutation_cluster_intervals');write(pd.DataFrame([bootstrap_effect(ab)]),'temperature_ablation_paired_interval')
    write(ab.groupby('fold').agg(full_mae=('base_abs_error','mean'),without_temperature_mae=('altered_abs_error','mean')).reset_index(),'temperature_ablation_fold_metrics')

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['metadata','controls','temperature_controls','importance','all']);a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    for stage,fn in [('metadata',metadata),('controls',controls),('temperature_controls',lambda:controls(True)),('importance',importance)]:
        if a.stage in [stage,'all']:fn()

if __name__=='__main__':main()
