#!/usr/bin/env python3
"""Summarize saved final SCOTS base-case flows without additional optimization."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import csv,json,math,statistics
R=Path(__file__).resolve().parents[2]/'check/results'
P='scots_exp5_scopf_warmstart_native_20260927'
def one(row):
 d=json.loads(Path(row['solution_file']).read_text());sol=d['result']['solution']['nw']['1']['branch']; params=d['branch_parameters']
 c=set(d['metadata'].get('generation_metadata',{}).get('coupling_branch_rows_zero_based',[]))
 coupling=[]; other=[]; utilization=[]; switches=0
 for k,b in params.items():
  v=sol[k];f=max(abs(v['pf']),abs(v['pt']))*d['base_mva']; assert math.isfinite(f)
  original=int(b['source_id'][1])-1
  (coupling if original in c else other).append(f)
  if b.get('rate_a',0)>0: utilization.append(abs(v['pf'])/b['rate_a'])
  if original in c and v.get('br_status',1)<0.5:switches+=1
 assert len(coupling)==len(c)
 ordered=sorted(coupling); pos=.95*(len(ordered)-1); low=math.floor(pos); high=math.ceil(pos)
 q95=ordered[low]+(pos-low)*(ordered[high]-ordered[low])
 return {'state':'complete','scopf_feasible':True,'expected_coupling_branch_count':len(coupling),'positive_flow_branch_count':sum(x>1e-6 for x in coupling),'coupling_branch_observation_fraction':1.0,'noncoupling_branch_observation_fraction':1.0,'coupling_mean_absolute_flow_mw':statistics.mean(coupling),'coupling_p95_absolute_flow_mw':q95,'coupling_maximum_absolute_flow_mw':max(coupling),'noncoupling_mean_absolute_flow_mw':statistics.mean(other),'artifact_stem':row['artifact_stem'],'strategy':row['strategy'],'coupling_count':len(coupling),'positive_count':sum(x>1e-6 for x in coupling),'switched_off_coupling':switches,'coupling_mean':statistics.mean(coupling) if coupling else None,'noncoupling_mean':statistics.mean(other),'max_utilization':max(utilization),'overload_cost':d['objective_costs'].get('overload_all'),'optimal':row['termination_status']=='OPTIMAL'}
def main():
 rows=list(csv.DictReader((R/f'{P}_generated/summary.csv').open())); rows=[x for x in rows if x['outcome']=='feasible']
 with ThreadPoolExecutor(max_workers=4) as pool: data=list(pool.map(one,rows))
 with (R/f'{P}_generated/flow_summary.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 for method in ['random','high_degree','kamada_kawai','heuristic_outer_face','exact_outer_face']:
  a=[x for x in data if x['strategy']==method]
  print(method,'n',len(a),'coupling',sum(x['coupling_count'] for x in a),'positive',sum(x['positive_count'] for x in a),'switched_off',sum(x['switched_off_coupling'] for x in a),'medians', {k:statistics.median(x[k] for x in a) for k in ['coupling_mean','noncoupling_mean','max_utilization']},flush=True)
 print('TOTAL',len(data),'coupling',sum(x['coupling_count'] for x in data),'positive',sum(x['positive_count'] for x in data),'switched_off',sum(x['switched_off_coupling'] for x in data),flush=True)
if __name__=='__main__':main()
