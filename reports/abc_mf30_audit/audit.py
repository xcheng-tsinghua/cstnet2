from pathlib import Path
import tempfile, json, csv, heapq
import h5py
import numpy as np

ROOT=Path(tempfile.gettempdir())/'cstnet2-abc-mf30-audit'
OUT=Path('reports/abc_mf30_audit'); OUT.mkdir(parents=True,exist_ok=True)
TYPES=['plane','cylinder','cone','sphere','other']
columns={k:[] for k in ['pmt','locnorm','dim','residual','canonical','dirnorm']}
examples=[]; cloud_rows=[]; shards=[]; nonfinite={}; xyz_min=np.full(3,np.inf); xyz_max=-xyz_min.copy()
trim_totals={str(r):[0.,0] for r in [0,.01,.05,.10]}

def stats(values):
 values=np.asarray(values,dtype=np.float64); values=values[np.isfinite(values)]
 if not values.size:return {'count':0}
 q=np.quantile(values,[.5,.9,.95,.99,.999])
 return dict(count=int(values.size),mean=float(values.mean()),median=float(q[0]),p90=float(q[1]),p95=float(q[2]),p99=float(q[3]),p999=float(q[4]),max=float(values.max()),min=float(values.min()))

def evaluate(x,p,n,d,l):
 norm=np.linalg.norm(n,axis=1); a=n/np.maximum(norm[:,None],1e-15)
 delta=x-l; z=np.einsum('ij,ij->i',delta,a)
 radial=np.linalg.norm(delta-z[:,None]*a,axis=1)
 res=np.full(len(x),np.nan); canonical=np.full(len(x),np.nan)
 m=p==0; res[m]=np.abs(z[m]); canonical[m]=np.linalg.norm(l[m]-np.sum(l[m]*a[m],axis=1)[:,None]*a[m],axis=1)
 m=p==1; res[m]=np.abs(radial[m]-d[m]); canonical[m]=np.abs(np.sum(l[m]*a[m],axis=1))
 m=p==2; res[m]=np.abs(radial[m]*np.cos(d[m])-np.abs(z[m])*np.sin(d[m]))
 m=p==3; res[m]=np.abs(np.linalg.norm(delta[m],axis=1)-d[m])
 return norm,res,canonical

for si in [0,13,26,33,46,59,67]:
 path=ROOT/f'stage1-{si:05d}-of-00068.h5'
 with h5py.File(path,'r') as f:
  offsets=f['offsets'][:]; ids=f['source_path'][:] if 'source_path' in f else None
  print('analyzing',path.name,'samples',len(offsets)-1,'fields',list(f),flush=True)
  data={k:f[k][:] for k in ['xyz','pmt','direction','dimension','location','affiliate_idx']}
  for k,v in data.items():nonfinite[k]=nonfinite.get(k,0)+int((~np.isfinite(v)).sum())
  x=data['xyz'].astype('float64'); p=data['pmt'].reshape(-1); n=data['direction'].astype('float64'); d=data['dimension'].reshape(-1).astype('float64'); l=data['location'].astype('float64'); aff=data['affiliate_idx'].reshape(-1)
  norms,res,canonical=evaluate(x,p,n,d,l); ln=np.linalg.norm(l,axis=1)
  xyz_min=np.minimum(xyz_min,x.min(axis=0)); xyz_max=np.maximum(xyz_max,x.max(axis=0))
  for key,value in dict(pmt=p,locnorm=ln,dim=d,residual=res,canonical=canonical,dirnorm=norms).items():columns[key].append(value.copy())
  shard={'file':path.name,'samples':len(offsets)-1,'points':len(x),'zero_loc_baseline':stats(ln[p<4]),'residual':stats(res),'gt_histogram':np.bincount(p,minlength=5).tolist()}; shards.append(shard)
  for j,(start,stop) in enumerate(zip(offsets[:-1],offsets[1:])):
   start=int(start);stop=int(stop); sl=slice(start,stop); pc=p[sl]; lc=ln[sl]
   valid=lc[(pc>=0)&(pc<4)&np.isfinite(lc)]
   source=str(j) if ids is None else (ids[j].decode() if isinstance(ids[j],bytes) else str(ids[j]))
   row={'shard':si,'sample_index':j,'sample_id':source,'points':stop-start,'xyz_absmax':float(np.abs(x[sl]).max()),'zero_loc_mean':float(valid.mean()) if len(valid) else 0.,'max_locnorm':float(lc.max()),'max_surface_residual':float(np.nanmax(res[sl])) if np.isfinite(res[sl]).any() else 0.}
   cloud_rows.append(row)
   ordered=np.sort(valid)
   for ratio in [0,.01,.05,.10]:
    keep=len(ordered)-int(len(ordered)*ratio); trim_totals[str(ratio)][0]+=float(ordered[:keep].sum());trim_totals[str(ratio)][1]+=keep
   if len(valid) and (row['max_locnorm']>5 or row['max_surface_residual']>.01):
    for aid in np.unique(aff[sl]):
     idx=np.flatnonzero(aff[sl]==aid)+start; k=int(idx[0]); t=int(p[k])
     if t==4 or (ln[idx].max()<=5 and np.nanmax(res[idx])<=.01):continue
     ex={'shard':si,'sample_index':j,'sample_id':source,'affiliate_idx':int(aid),'type':TYPES[t],'point_count':len(idx),'loc':l[k].tolist(),'locnorm':float(ln[k]),'dir':n[k].tolist(),'dim':float(d[k]),'xyz_min':x[idx].min(axis=0).tolist(),'xyz_max':x[idx].max(axis=0).tolist(),'surface_residual_mean':float(np.nanmean(res[idx])),'surface_residual_max':float(np.nanmax(res[idx])),'canonical_error':float(canonical[k]) if np.isfinite(canonical[k]) else None,'instance_loc_spread':float(np.linalg.norm(l[idx]-l[k],axis=1).max())}
     examples.append(ex)
  print(json.dumps(shard),flush=True)
  del data,x,p,n,d,l,aff,norms,res,canonical,ln

allv={k:np.concatenate(v) for k,v in columns.items()};p=allv['pmt'];valid=(p>=0)&(p<4);ln=allv['locnorm'][valid];sumloc=ln.sum()
summary={'dataset':'ZXCCHENGXI/cstnet2_s1_abc_mf30','sampling':'All points and samples in shards 0, 13, 26, 33, 46, 59, 67 (spread across repository; not an IID random sample).','shards':shards,'sample_count':len(cloud_rows),'point_count':len(p),'xyz_min':xyz_min.tolist(),'xyz_max':xyz_max.tolist(),'nonfinite':nonfinite,'zero_prediction_location_baseline':stats(ln),'zero_prediction_per_cloud_trim_means':{k:v[0]/max(v[1],1) for k,v in trim_totals.items()},'by_type':{}}
for t,name in enumerate(TYPES):
 m=p==t; v=allv['locnorm'][m]
 entry={'points':int(m.sum()),'point_fraction':float(m.mean()),'loc_norm':stats(v),'dimension':stats(allv['dim'][m]),'direction_norm':stats(allv['dirnorm'][m]),'surface_residual':stats(allv['residual'][m]),'canonical_error':stats(allv['canonical'][m]),'share_of_valid_loc_norm_sum':float(v.sum()/sumloc) if t<4 else None,'zero_loc_mse':float(np.square(v).mean()/3) if len(v) else None,'share_of_valid_loc_squared_sum':float(np.square(v).sum()/np.square(ln).sum()) if t<4 else None,'locnorm_above':{str(th):int((v>th).sum()) for th in [1,2,5,10,100,1000]}}
 summary['by_type'][name]=entry
summary['zero_loc_mse']=float(np.square(ln).mean()/3)
summary['large_loc_tail']={str(th):{'point_count':int((ln>th).sum()),'point_fraction':float((ln>th).mean()),'share_of_loc_norm_sum':float(ln[ln>th].sum()/sumloc),'share_of_loc_squared_sum':float(np.square(ln[ln>th]).sum()/np.square(ln).sum())} for th in [1,2,5,10,100,1000]}
summary['top_loc_instances']=sorted(examples,key=lambda v:v['locnorm'],reverse=True)[:30]
summary['top_residual_instances']=sorted(examples,key=lambda v:v['surface_residual_max'],reverse=True)[:30]
summary['top_clouds']=sorted(cloud_rows,key=lambda v:v['zero_loc_mean'],reverse=True)[:30]
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
with (OUT/'clouds.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=cloud_rows[0].keys());w.writeheader();w.writerows(cloud_rows)
print(json.dumps({k:v for k,v in summary.items() if k not in ['top_loc_instances','top_residual_instances','top_clouds']},indent=2),flush=True)
print('saved',str(OUT.resolve()),flush=True)


