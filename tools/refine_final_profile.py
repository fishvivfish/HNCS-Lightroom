#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from hncs_core.adobe_sdk_color import TripleColorSpec, adobe_line_slots
from hncs_core.adobe_triple_illuminant import IlluminantDescriptor, temperature_tint_to_xy
from tools.carrier_model import Experiment
from tools.hncs_full_probe import FullProbe
from tools.optimize_carrier import DescriptorProblem

DEFAULT_ORIGINAL_DCP=ROOT/'local_assets'/'Sony ILCE-7RM5 Adobe Standard.dcp'
DEFAULT_ACTIVE=ROOT/'config'/'final_active_set.json'
DEFAULT_OUTPUT=ROOT/'work'/'final_refinement'
# Numerical initialization retained from the converged optimization basin; not physical calibration constants.
TINTS=np.array([22.569588233882797,-44.65722160623658,7.210552508117358],float)
START_T=np.array([2411.46,6821.38,16814.83],float)
EPS=0.004

def descriptors(x6):
    return tuple(IlluminantDescriptor(float(x6[i]),float(x6[i+3])) for i in range(3))

def project_c(exp,x6,temps,c0=None,max_iter=20):
    D=descriptors(x6); c=np.array([0.95,0.3,0.0] if c0 is None else c0,float)
    g=np.array([exp.original.set_white_xy(temperature_tint_to_xy(float(T),0.0)).weights[0] for T in temps])
    for _ in range(max_iter):
        colors,forwards=adobe_line_slots(exp.cm1,exp.cm2,exp.fm1,exp.fm2,c)
        spec=TripleColorSpec(D,colors,forwards)
        W=[]
        for T in temps:
            os=exp.original.set_white_xy(temperature_tint_to_xy(float(T),0.0))
            cw=spec.neutral_to_xy(os.camera_white); W.append(spec.set_white_xy(cw).weights)
        W=np.asarray(W)
        nc=lsq_linear(W,g,bounds=(0.,1.),lsq_solver='exact',tol=1e-12,max_iter=100).x
        if np.max(np.abs(nc-c))<1e-10: c=nc; break
        c=nc
    return c

def build_ideals(exp,values,temps,folder,iters=10):
    folder.mkdir(parents=True,exist_ok=True)
    probe=FullProbe(exp,values)
    ideals=[]; timing={}
    for i,T in enumerate(temps,1):
        cp=folder/f'ideal_{int(T)}K.npz'
        t0=time.time()
        if cp.exists():
            p=np.asarray(np.load(cp)['payload'],dtype=np.float32); dt=time.time()-t0; source='cache'
        else:
            p=probe.ideal_payload(int(T),iters=iters); dt=time.time()-t0; source='built'; np.savez_compressed(cp,payload=p)
        ideals.append(p.astype(np.float32)); timing[str(int(T))]=dt
        print(f'ideal {i}/{len(temps)} {int(T)}K {source} {dt:.2f}s',flush=True)
    arr=np.stack(ideals)
    np.savez_compressed(folder/'ideals.npz',temperatures_K=temps,ideals=arr,values=values)
    return arr,timing

def clean_result(r):
    return {k:v for k,v in r.items() if k!='_bases'}

def pattern_temp(problem,x6,steps=(80.,160.,800.),min_steps=(0.5,1.,5.),max_evals=90):
    x=x6.copy(); cur=problem.evaluate(x,full_base=True)
    if not cur['feasible']: raise RuntimeError(f'start infeasible base={cur["adobe_base_worst_p95"]}')
    best=float(cur['hncs']['max_p95']); evals=1; step=np.array(steps,float); hist=[]
    bounds=np.array([[1667.,5000.],[3000.,8000.],[6000.,25000.]])
    print('pattern start',best,cur['adobe_base_worst_p95'],x[:3],flush=True)
    while evals<max_evals and np.any(step>np.array(min_steps)):
        improved=False
        for j in range(3):
            candidates=[]
            for sign in (-1.,1.):
                y=x.copy(); y[j]=np.clip(y[j]+sign*step[j],bounds[j,0],bounds[j,1])
                r=problem.evaluate(y,full_base=True); evals+=1
                score=float(r['hncs']['max_p95']) if r['feasible'] else np.inf
                candidates.append((score,y,r))
                print('probe',evals,'j',j,'T',y[:3].tolist(),'base',r['adobe_base_worst_p95'],'hncs',None if not r['feasible'] else score,flush=True)
            score,y,r=min(candidates,key=lambda q:q[0])
            if score+1e-8 < best:
                x=y; cur=r; best=score; improved=True
                hist.append({'eval':evals,'best':best,'base':cur['adobe_base_worst_p95'],'x6':x.tolist()})
                print('ACCEPT',best,cur['adobe_base_worst_p95'],x[:3],flush=True)
        if not improved:
            step*=0.5; print('halve',step,flush=True)
    return x,cur,hist,evals

def tint_probe(problem,x6,steps=(4.,4.,4.)):
    base=problem.evaluate(x6,full_base=True); best=base['hncs']['max_p95']; bx=x6.copy(); br=base
    records=[]
    for j in range(3):
        for sign in (-1.,1.):
            y=bx.copy(); y[3+j]+=sign*steps[j]
            r=problem.evaluate(y,full_base=True); score=r['hncs']['max_p95'] if r['feasible'] else np.inf
            records.append({'var':j,'value':float(y[3+j]),'base':float(r['adobe_base_worst_p95']),'hncs':None if not r['feasible'] else float(score)})
            if score+5e-5 < best: best=score; bx=y; br=r
    return bx,br,records

def main():
    parser=argparse.ArgumentParser(description="Run the final adaptive HNCS Color refinement from local assets.")
    parser.add_argument('--original-dcp',type=Path,default=DEFAULT_ORIGINAL_DCP)
    parser.add_argument('--active-set',type=Path,default=DEFAULT_ACTIVE)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--ideal-iters',type=int,default=10)
    args=parser.parse_args()
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    active=json.loads(args.active_set.read_text()); temps=np.array(active['active_temperatures_K'],dtype=np.int32)
    exp=Experiment(args.original_dcp,sample_count=40000)
    x6=np.r_[START_T,TINTS]; c=project_c(exp,x6,temps); values=np.r_[x6,c]
    report={'classification':'FINAL ADAPTIVE REFINEMENT WITH EXACT IDEAL-HSM REFRESH','start_x6':x6.tolist(),'cycles':[]}
    for cycle in range(1,4):
        print('\n=== REFRESH',cycle,'===',flush=True)
        ideals,timing=build_ideals(exp,values,temps,out/f'refresh_{cycle}',iters=args.ideal_iters)
        problem=DescriptorProblem(exp,values,temps,ideals,base_samples=40000,epsilon=EPS)
        exact=problem.evaluate(x6,full_base=True)
        print('REFRESH EVAL',json.dumps({'base':exact['adobe_base_worst_p95'],'hncs':exact.get('hncs'),'values':exact['values']},indent=2),flush=True)
        xnew,r,hist,ne=pattern_temp(problem,x6,max_evals=90)
        selected=np.asarray(r['values'],float)
        report['cycles'].append({'cycle':cycle,'refresh_values':values.tolist(),'refresh_eval':clean_result(exact),'optimized_x6':xnew.tolist(),'optimized_eval':clean_result(r),'history':hist,'evals':ne,'timing':timing})
        (out/'progress.json').write_text(json.dumps(report,indent=2))
        move=np.max(np.abs(xnew[:3]-x6[:3])); improve=float(exact['hncs']['max_p95']-r['hncs']['max_p95'])
        x6=xnew; values=selected
        print('cycle move',move,'frozen improve',improve,flush=True)
        if cycle>=2 and move<3.0 and improve<2e-5: break
    print('\n=== FINAL EXACT REFRESH ===',flush=True)
    ideals,timing=build_ideals(exp,values,temps,out/'final_refresh',iters=args.ideal_iters)
    problem=DescriptorProblem(exp,values,temps,ideals,base_samples=40000,epsilon=EPS)
    final_exact=problem.evaluate(x6,full_base=True)
    tx,tr,tprobe=tint_probe(problem,x6); tint_adopt=False
    if np.max(np.abs(tx[3:]-x6[3:]))>0:
        tvalues=np.asarray(tr['values'],float); tideals,_=build_ideals(exp,tvalues,temps,out/'tint_refresh',iters=args.ideal_iters)
        tp=DescriptorProblem(exp,tvalues,temps,tideals,base_samples=40000,epsilon=EPS); texact=tp.evaluate(tx,full_base=True)
        if texact['feasible'] and texact['hncs']['max_p95']+5e-5 < final_exact['hncs']['max_p95']:
            tint_adopt=True; x6=tx; values=np.asarray(texact['values'],float); final_exact=texact; ideals=tideals
        report['tint_exact_check']=clean_result(texact)
    report['tint_probe']=tprobe; report['tint_adopted']=tint_adopt; report['final_x6']=x6.tolist(); report['final_values']=np.asarray(final_exact['values'],float).tolist(); report['final_exact']=clean_result(final_exact); report['final_timing']=timing
    np.savez_compressed(out/'final_active_solution.npz',temperatures_K=temps,ideals=ideals,bases=final_exact['_bases'],values=np.asarray(final_exact['values'],float))
    (out/'final_refinement_report.json').write_text(json.dumps(report,indent=2))
    print('\nFINAL REFINEMENT',json.dumps({'x6':report['final_x6'],'values':report['final_values'],'base':final_exact['adobe_base_worst_p95'],'hncs':final_exact['hncs'],'tint_adopted':tint_adopt},indent=2),flush=True)
if __name__=='__main__': main()
