import numpy as np, h5py, json,tempfile
from pathlib import Path
out=[]
with h5py.File(Path(tempfile.gettempdir())/'cstnet2-abc-mf30-audit/stage1-00033-of-00068.h5','r') as f:
 for j,aid in [(1282,24),(1072,2),(556,8)]:
  s,e=map(int,f['offsets'][j:j+2]); aff=f['affiliate_idx'][s:e]; m=aff==aid
  x=f['xyz'][s:e][m].astype('float64'); a=f['direction'][s:e][m][0].astype('float64'); a/=np.linalg.norm(a)
  l=f['location'][s:e][m][0].astype('float64'); angle=float(f['dimension'][s:e][m][0]); z=x@a
  foot=l-(l@a)*a; r=np.linalg.norm(x-foot-z[:,None]*a,axis=1)
  slope,b=np.linalg.lstsq(np.column_stack((z,np.ones(len(z)))),r,rcond=None)[0]
  pred_apex=-b/slope
  true_res=np.abs(r*np.cos(angle)-np.abs(z-l@a)*np.sin(angle))
  fitted_res=np.abs(r-(slope*z+b))/np.sqrt(1+slope*slope)
  cylinder_res=np.abs(r-r.mean())
  out.append(dict(sample_index=j,affiliate_idx=aid,gt_apex_axis_coordinate=float(l@a),gt_angle_deg=float(np.rad2deg(angle)),gt_surface_residual_mean=float(true_res.mean()),gt_surface_residual_max=float(true_res.max()),oracle_axis_cone_fit_apex_coordinate=float(pred_apex),oracle_axis_cone_fit_apex_error=float(abs(pred_apex-l@a)),oracle_axis_cone_fit_residual_mean=float(fitted_res.mean()),same_axis_cylinder_residual_mean=float(cylinder_res.mean()),same_axis_cylinder_radius=float(r.mean()),radial_range=float(np.ptp(r)),axial_range=float(np.ptp(z))))
Path('reports/abc_mf30_audit/cone_fit_diagnostic.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
