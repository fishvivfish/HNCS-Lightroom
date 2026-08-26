#!/usr/bin/env python3
"""Exact WB-dependent HNCS target and HDR HueSatMap probe used by the final solver.

Scientific constants in this module are the frozen final definitions:
72 x 32 x 32 HSM, EV -3..+3, and p=0.53. Recovered numerical HNCS assets
are loaded from local_assets/phocus401_hncs_color and are intentionally not
bundled by this public source tree.
"""
from pathlib import Path
import json, sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from hncs_core.colour import (
    D50_WHITE_XY, PROPHOTO_PRIMARIES, rgb_to_xyz_matrix,
    prophoto_to_xyz_d65, xyz_d65_to_oklab,
)
from hncs_core.phocus401_wb import Phocus401WBModel
from hncs_core.color_correct import apply_color_correct
from hncs_core.adobe_triple_illuminant import temperature_tint_to_xy
from hncs_core.dng_huesat import GridSpec, rgb_to_hsv, hsv_to_rgb, apply_payload_rgb

H,S,V=72,32,32
MAX_USED_V=16
HSM_SPEC=GridSpec(H,S,V)
P=0.53
STOPS=(-3,-2,-1,0,1,2,3)
SCALES=np.asarray([2.0**e for e in STOPS],dtype=np.float64)
WEIGHTS=SCALES**(-P)


def enc_over(x):
    x=np.maximum(np.asarray(x,dtype=np.float64),0.0)
    return x*(256.0+x)/(256.0*(1.0+x))

def dec_over(x):
    x=np.clip(np.asarray(x,dtype=np.float64),0.0,1.0)
    return 16.0*((8.0*x)-8.0+np.sqrt(np.maximum(64.0*x*x-127.0*x+64.0,0.0)))

def apply_map_hdr(spec:GridSpec,payload:np.ndarray,rgb:np.ndarray)->np.ndarray:
    x=np.maximum(np.asarray(rgb,dtype=np.float64),0.0)
    work=enc_over(x) if spec.value>1 else x
    # interpolation on encoded domain
    hsv=rgb_to_hsv(work)
    hue_scaled=hsv[:,0]*spec.hue/6.0
    sat_scaled=np.clip(hsv[:,1],0,1)*(spec.saturation-1)
    hbase=np.floor(hue_scaled); h0=hbase.astype(np.int64)%spec.hue; h1=(h0+1)%spec.hue; hf=hue_scaled-hbase
    s0=np.minimum(np.floor(sat_scaled).astype(np.int64),spec.saturation-2); s1=s0+1; sf=sat_scaled-s0
    if spec.value>1:
        val_scaled=np.clip(hsv[:,2],0,1)*(spec.value-1)
        v0=np.minimum(np.floor(val_scaled).astype(np.int64),spec.value-2); v1=v0+1; vf=val_scaled-v0
    else:
        v0=v1=np.zeros(len(hsv),dtype=np.int64); vf=np.zeros(len(hsv))
    grid=np.asarray(payload,dtype=np.float64).reshape(spec.value,spec.hue,spec.saturation,3)
    d=np.zeros((len(hsv),3),dtype=np.float64)
    for vi,vw in ((v0,1-vf),(v1,vf)):
        for hi,hw in ((h0,1-hf),(h1,hf)):
            for si,sw in ((s0,1-sf),(s1,sf)):
                d += grid[vi,hi,si]*(vw*hw*sw)[:,None]
    out=hsv.copy(); out[:,0]=hsv[:,0]+d[:,0]/60.0; out[:,1]=np.clip(hsv[:,1]*d[:,1],0,1); out[:,2]=np.clip(hsv[:,2]*d[:,2],0,1)
    y=hsv_to_rgb(out)
    return dec_over(y) if spec.value>1 else y


def payload_hdr(h:int,s:int,v:int,mapping,neutral_identity:bool=False)->np.ndarray:
    hsv=np.asarray([(hi*6.0/h,si/(s-1),vi/(v-1)) for vi in range(v) for hi in range(h) for si in range(s)],dtype=np.float64)
    src=dec_over(hsv_to_rgb(hsv))
    dst=np.maximum(mapping(src),0.0)
    dh=rgb_to_hsv(enc_over(dst))
    si=np.tile(np.arange(s),h*v); vi=np.repeat(np.arange(v),h*s)
    hue=((dh[:,0]-hsv[:,0]+3.0)%6.0-3.0)*60.0; hue[dh[:,1]<1e-11]=0.0
    sat=np.divide(dh[:,1],hsv[:,1],out=np.ones(len(hsv)),where=hsv[:,1]>1e-12); sat=np.clip(sat,0,256)
    val=np.divide(dh[:,2],hsv[:,2],out=np.ones(len(hsv)),where=hsv[:,2]>1e-12); val=np.clip(val,0,256)
    p=np.stack((hue,sat,val),axis=-1)
    p[vi==0]=(0,1,1)
    if neutral_identity: p[si==0]=(0,1,1)
    else:
        p[si==0,0]=0; p[si==0,1]=1
    return p.reshape(v,h,s,3).astype(np.float32)


class HNCSProjector:
    def __init__(self,model_dir:Path):
        self.model=Phocus401WBModel.load(model_dir)
        working=json.loads((ROOT/'config'/'hasselblad_rgb.json').read_text(encoding='utf-8'))
        h_to_xyz=np.asarray(working['rgb_to_xyz'],dtype=np.float64)
        self.xyz_to_h=np.linalg.inv(h_to_xyz)
        self.h_to_xyz=h_to_xyz
        self.pro_to_xyz=rgb_to_xyz_matrix(PROPHOTO_PRIMARIES,D50_WHITE_XY)
        self.xyz_to_pro=np.linalg.inv(self.pro_to_xyz)
    def target(self,rgb:np.ndarray,temperature:int)->np.ndarray:
        x=np.asarray(rgb,dtype=np.float64).reshape(-1,3)
        xyz=x@self.pro_to_xyz.T
        # Exact recovered camera matrix at Kelvin, used only for ColorCorrect gray/chroma weighting.
        k=int(temperature); mk=min(max(k,2000),8000)
        camM=self.model._piecewise(self.model.camera_matrices,self.model.matrix_temperatures,mk)
        pseudo_cam=xyz@np.linalg.inv(camM).T
        hb=xyz@self.xyz_to_h.T
        params,tex=self.model.parameters_at_kelvin(float(temperature))
        p=dict(params); p['input_matrix']=np.asarray(self.model.rgb_to_ycc,dtype=np.float64)
        corrected=apply_color_correct(hb,p,tex,clip_input=False,weighting_rgb=pseudo_cam)
        return (corrected@self.h_to_xyz.T)@self.xyz_to_pro.T


def _stats(x):
    x=np.asarray(x,float); return {'median':float(np.median(x)),'p95':float(np.percentile(x,95)),'max':float(np.max(x))}

def met(a,c):
    a=np.asarray(a,float); c=np.asarray(c,float)
    er=np.linalg.norm(a-c,axis=1)
    de=np.linalg.norm(xyz_d65_to_oklab(prophoto_to_xyz_d65(a))-xyz_d65_to_oklab(prophoto_to_xyz_d65(c)),axis=1)
    return {'rgb':_stats(er),'oklab':_stats(de)}

def test_grid(seed:int=260237)->np.ndarray:
    rng=np.random.default_rng(seed); rand=rng.random((729,3))
    hsv=np.asarray([(hi*6/9,si/8,vi/8) for vi in range(9) for hi in range(9) for si in range(9)],float)
    return np.vstack([rand,hsv_to_rgb(hsv)])


def _source_nodes_used():
    hsv=np.asarray([(hi*6.0/H,si/(S-1),vi/(V-1)) for vi in range(MAX_USED_V+1) for hi in range(H) for si in range(S)],float)
    return hsv, dec_over(hsv_to_rgb(hsv))

SRC_HSV,SRC_X=_source_nodes_used()


class FullProbe:
    def __init__(self,experiment,values):
        self.exp=experiment
        self.values=np.asarray(values,dtype=np.float64)
        self.spec,_,_,_=experiment.candidate(self.values)
        self.projector=HNCSProjector(ROOT/'local_assets'/'phocus401_hncs_color')
        self._build_downstream()
    def _build_downstream(self):
        # Adobe's original LookTable, with homogeneous extension above 1.
        def adobe_sdr(x): return apply_payload_rgb(self.exp.look_spec,self.exp.look,np.clip(np.asarray(x,float),0,1))
        self._adobe_sdr=adobe_sdr
        self.ADOBE_SPEC=GridSpec(108,32,24)
        self.ADOBE_HDR=payload_hdr(108,32,24,self.adobe_look_extended,neutral_identity=False)
        self.FIXED_SPEC=GridSpec(144,64,24)
        self.FIXED_HDR=payload_hdr(144,64,24,lambda x:self.projector.target(x,5550),neutral_identity=False)
    def adobe_look_extended(self,x):
        x=np.maximum(np.asarray(x,float),0); scale=np.maximum(np.max(x,axis=1),1.0); norm=x/scale[:,None]
        return self._adobe_sdr(norm)*scale[:,None]
    def adobe_hdr(self,x): return apply_map_hdr(self.ADOBE_SPEC,self.ADOBE_HDR,x)
    def fixed5550(self,x): return apply_map_hdr(self.FIXED_SPEC,self.FIXED_HDR,x)
    def downstream(self,z): return self.fixed5550(self.adobe_hdr(z))
    def _states(self,T:int):
        white=temperature_tint_to_xy(float(T),0.0); os=self.exp.original.set_white_xy(white)
        cw=self.spec.neutral_to_xy(os.camera_white); cs=self.spec.set_white_xy(cw)
        return os,cs
    def _candidate_to_original_pro(self,x,T:int):
        os,cs=self._states(T)
        # row-vector cameraRGB -> ProPhoto maps
        Mc=cs.camera_to_pcs.T@self.exp.prophoto_from_pcs.T
        Mo=os.camera_to_pcs.T@self.exp.prophoto_from_pcs.T
        return np.asarray(x,float)@np.linalg.inv(Mc)@Mo
    def original_post_hsm(self,x,T:int):
        xo=np.maximum(self._candidate_to_original_pro(x,T),0.0)
        os,_=self._states(T); w=self.exp.original.weights(os.white_xy)
        hsm=(w[0]*self.exp.original_hsm[0]+w[1]*self.exp.original_hsm[1]).astype(np.float32)
        return apply_payload_rgb(self.exp.base_hsm_spec,hsm,np.clip(xo,0,1))
    def original_target(self,x,T:int,ev:int=0):
        a=self.original_post_hsm(x,T)
        return self.projector.target(self.adobe_look_extended(np.maximum(a,0)*(2.0**ev)),int(T))
    def candidate_output(self,x,T:int,payload:np.ndarray,ev:int=0):
        z=apply_map_hdr(HSM_SPEC,payload,np.clip(np.asarray(x,float),0,1))
        return self.downstream(z*(2.0**ev))
    def _weighted_invert(self,targets,z0,iters=10,eps=2e-5):
        z=np.clip(np.asarray(z0,float),0,4); targets=[np.asarray(q,float) for q in targets]; n=len(z)
        for _ in range(iters):
            Fs=[self.downstream(z*s) for s in SCALES]
            R=np.concatenate([WEIGHTS[j]*(targets[j]-Fs[j]) for j in range(len(SCALES))],axis=1)
            J=np.empty((n,3*len(SCALES),3),float)
            for cc in range(3):
                zp=z.copy(); zm=z.copy(); zp[:,cc]+=eps; zm[:,cc]=np.maximum(zm[:,cc]-eps,0); den=zp[:,cc]-zm[:,cc]
                for j,s in enumerate(SCALES):
                    fp=self.downstream(zp*s); fm=self.downstream(zm*s)
                    J[:,3*j:3*j+3,cc]=WEIGHTS[j]*(fp-fm)/np.maximum(den[:,None],1e-12)
            JT=np.swapaxes(J,1,2); A=JT@J+2e-6*np.eye(3)[None]; q=(JT@R[...,None])[...,0]
            try: d=np.linalg.solve(A,q[...,None])[...,0]
            except np.linalg.LinAlgError: d=np.einsum('nij,nj->ni',np.linalg.pinv(A),q)
            scl=np.maximum(1,np.max(np.abs(d),axis=1)/0.15); d/=scl[:,None]; z=np.clip(z+d,0,4)
        return z
    def ideal_payload(self,T:int,iters=10):
        a=self.original_post_hsm(SRC_X,int(T))
        targets=[self.projector.target(self.adobe_look_extended(np.maximum(a,0)*s),int(T)) for s in SCALES]
        z=self._weighted_invert(targets,a,iters=iters)
        zh=rgb_to_hsv(enc_over(z)); src=SRC_HSV
        hue=((zh[:,0]-src[:,0]+3)%6-3)*60; hue[zh[:,1]<1e-11]=0
        sat=np.divide(zh[:,1],src[:,1],out=np.ones(len(src)),where=src[:,1]>1e-12); sat=np.clip(sat,0,256)
        val=np.divide(zh[:,2],src[:,2],out=np.ones(len(src)),where=src[:,2]>1e-12); val=np.clip(val,0,256)
        vals=np.stack((hue,sat,val),-1).astype(np.float32)
        p=np.zeros((V,H,S,3),np.float32); p[...,1:]=1
        row=0
        for vv in range(MAX_USED_V+1):
            for hh in range(H):
                for ss in range(S): p[vv,hh,ss]=vals[row]; row+=1
        p[0,:,:]=(0,1,1); p[:MAX_USED_V+1,:,0,0]=0; p[:MAX_USED_V+1,:,0,1]=1
        for vv in range(MAX_USED_V+1,V): p[vv]=p[MAX_USED_V]
        return p