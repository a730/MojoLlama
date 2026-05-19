#!/usr/bin/env python3
"""CEngine — pure C forward pass. Uses ctypes.Structure for the EC struct."""
import sys,os,ctypes,numpy as np,gguf
from gguf.constants import GGMLQuantizationType as QT
sys.path.insert(0,'/onedev-workspace/work/src/mojollama/kernels')
os.environ['PYTHONUNBUFFERED']='1'

kernel_dir='/onedev-workspace/work/src/mojollama/kernels'
BLS={2:18,3:20,12:144,13:176,14:210,8:34}

# EC struct matching c_engine.c
class EC(ctypes.Structure):
    _fields_ = [
        ("L",ctypes.c_int),("N",ctypes.c_int),("NH",ctypes.c_int),("NKH",ctypes.c_int),
        ("HD",ctypes.c_int),("FF",ctypes.c_int),("V",ctypes.c_int),("eps",ctypes.c_float),
        ("wQ",ctypes.c_void_p),("wK",ctypes.c_void_p),("wV",ctypes.c_void_p),
        ("wO",ctypes.c_void_p),("wG",ctypes.c_void_p),("wU",ctypes.c_void_p),("wD",ctypes.c_void_p),
        ("wAN",ctypes.c_void_p),("wFN",ctypes.c_void_p),
        ("nQ",ctypes.c_void_p),("nK",ctypes.c_void_p),("nV",ctypes.c_void_p),
        ("nO",ctypes.c_void_p),("nG",ctypes.c_void_p),("nU",ctypes.c_void_p),("nD",ctypes.c_void_p),
        ("nc",ctypes.c_int),
        ("emb",ctypes.c_void_p),("onw",ctypes.c_void_p),
        ("wOut",ctypes.c_void_p),("outNR",ctypes.c_int),("outNC",ctypes.c_int),
        ("kvK",ctypes.c_void_p),("kvV",ctypes.c_void_p),("kvL",ctypes.c_void_p),
        ("MP",ctypes.c_int),
        ("x",ctypes.c_void_p),("xn",ctypes.c_void_p),("res",ctypes.c_void_p),
        ("q",ctypes.c_void_p),("k",ctypes.c_void_p),("v",ctypes.c_void_p),
        ("att",ctypes.c_void_p),("gate",ctypes.c_void_p),("up",ctypes.c_void_p),
        ("silu",ctypes.c_void_p),("oproj",ctypes.c_void_p),("ffn",ctypes.c_void_p),
    ]

class CEngine:
    def __init__(self, model_path, n_threads=32):
        os.environ['OMP_NUM_THREADS']=str(n_threads)
        self.n_threads=n_threads
        self.lib=ctypes.CDLL(os.path.join(kernel_dir,'cengine.so'))
        self.lib.engine_forward.argtypes=[ctypes.c_int,ctypes.POINTER(EC),ctypes.c_void_p]
        self.lib.engine_forward.restype=None
        
        reader=gguf.GGUFReader(model_path)
        self._load_meta(reader)
        weights,raw_weights,weight_info=self._load_weights(reader)
        self._build_ec(weights,raw_weights,weight_info)
        print(f"CEngine: {self.L}L/{self.N}D/{self.FF}FF/{self.NH}H/{self.NKH}KV | t={n_threads}")
        self.pos=0
    
    def _load_meta(self,reader):
        f=reader.fields
        def g(n):
            for k,v in f.items():
                if k==n: d=v.parts[-1] if hasattr(v,'parts') and len(v.parts)>=1 else None
                if k==n and d is not None: return int(d[0]) if hasattr(d,'__iter__') and len(d)==1 else d
        arch='llama'
        for p in ['qwen3moe','qwen2moe','llama','mistral']:
            if any(f'{p}.block_count' in k for k in f): arch=p; break
        self.L=int(g(f'{arch}.block_count') or 16)
        self.N=int(g(f'{arch}.embedding_length') or 2048)
        self.FF=int(g(f'{arch}.feed_forward_length') or self.N*4)
        self.NH=int(g(f'{arch}.attention.head_count') or 32)
        self.NKH=int(g(f'{arch}.attention.head_count_kv') or self.NH)
        self.HD=int(g(f'{arch}.attention.key_length') or (self.N//self.NH))
        self.eps=float(g(f'{arch}.attention.layer_norm_rms_epsilon') or 1e-6)
        self.V=int(g(f'{arch}.vocab_size') or 0)
    
    def _load_weights(self,reader):
        w={}; rw={}; wi={}; CQT={0,2,3,8,12,13,14}
        for t in reader.tensors:
            n=t.name; qt=QT(t.tensor_type).value
            if len(t.shape)==2:
                id_,od=int(t.shape[0]),int(t.shape[1])
                f32=gguf.dequantize(t.data,t.tensor_type).astype(np.float32).reshape(od,id_)
                w[n]=np.ascontiguousarray(f32)
                if qt in CQT:
                    raw=np.ascontiguousarray(t.data.reshape(-1),dtype=np.uint8).copy()
                    rw[n]=raw; wi[n]=(od,id_,BLS.get(qt,0),qt)
            elif len(t.shape)==1:
                f32=gguf.dequantize(t.data,t.tensor_type).astype(np.float32).reshape(-1)
                w[n]=np.ascontiguousarray(f32)
        emb=w['token_embd.weight']
        if emb.ndim==2 and emb.shape[0]!=self.V: w['token_embd.weight']=np.ascontiguousarray(emb.T)
        if not self.V: self.V=w['token_embd.weight'].shape[0]
        return w,rw,wi
    
    def _requant(self,rw,info):
        nr,nc=info[0],info[1]; nb=nc//32
        omp=ctypes.CDLL(os.path.join(kernel_dir,'quant_kernels_omp.so'))
        cu=ctypes.POINTER(ctypes.c_uint8); cf=ctypes.POINTER(ctypes.c_float)
        omp.q6_k_dequantize_row.argtypes=[cu,cf,ctypes.c_int]; omp.q6_k_dequantize_row.restype=None
        f32=np.zeros((nr,nc),dtype=np.float32)
        rp=rw.ctypes.data_as(cu); ba=ctypes.addressof(rp.contents)
        for r in range(nr):
            ptr=ctypes.cast(ba+r*(nc//256)*210,cu)
            omp.q6_k_dequantize_row(ptr,f32[r].ctypes.data_as(cf),ctypes.c_int(nc))
        blks=f32.reshape(nr,nb,32); amax=np.max(np.abs(blks),axis=2,keepdims=True)
        amax=np.clip(amax,1e-10,None); d=amax/127.0
        qs=np.clip(np.round(blks/d),-127,127).astype(np.int8)
        dh=d[:,:,0].astype(np.float16).view(np.uint16)
        q8=np.zeros((nr,nb,34),dtype=np.uint8)
        q8[:,:,0]=dh&0xFF; q8[:,:,1]=(dh>>8)&0xFF; q8[:,:,2:]=qs.view(np.uint8)
        self._ob=np.ascontiguousarray(q8.reshape(-1),dtype=np.uint8)
        print(f"  Output Q8_0: {nr}×{nc} ({len(self._ob)/1024/1024:.0f} MB)")
        return self._ob.ctypes.data_as(ctypes.c_void_p), nr, nc
    
    def _build_ec(self,w,rw,wi):
        N=self.N; NKH=self.NKH; HD=self.HD; FF=self.FF; L=self.L; MP=4096
        
        # Per-layer arrays
        wQ,wK,wV,wO,wG,wU,wD=[],[],[],[],[],[],[]
        wQ,wK,wV,wO,wG,wU,wD=[],[],[],[],[],[],[]
        wAN,wFN=[],[]; nQ,nK,nV,nO,nG,nU,nD=[],[],[],[],[],[],[]
        wm={'wQ':wQ,'wK':wK,'wV':wV,'wO':wO,'wG':wG,'wU':wU,'wD':wD}
        nm={'nQ':nQ,'nK':nK,'nV':nV,'nO':nO,'nG':nG,'nU':nU,'nD':nD}
        out_n='output.weight' if 'output.weight' in w else 'token_embd.weight'
        for i in range(L):
            pfx=f'blk.{i}'
            for gn,wn,nn in [('attn_q','wQ','nQ'),('attn_k','wK','nK'),('attn_v','wV','nV'),
                            ('attn_output','wO','nO'),('ffn_gate','wG','nG'),
                            ('ffn_up','wU','nU'),('ffn_down','wD','nD')]:
                inf=wi.get(f'{pfx}.{gn}.weight')
                if inf:
                    wm[wn].append(rw[f'{pfx}.{gn}.weight'].ctypes.data_as(ctypes.c_void_p))
                    nm[nn].append(inf[0])
                else:
                    wm[wn].append(0); nm[nn].append(0)
            wAN.append(w[f'{pfx}.attn_norm.weight'].ctypes.data_as(ctypes.c_void_p))
            wFN.append(w[f'{pfx}.ffn_norm.weight'].ctypes.data_as(ctypes.c_void_p))
        
        # Output projection
        oi=wi.get(out_n)
        if oi and oi[3] in (6,14,12,10):
            wOut,outNR,outNC=self._requant(rw[out_n],oi)
        elif oi:
            wOut=rw[out_n].ctypes.data_as(ctypes.c_void_p); outNR,outNC=oi[0],oi[1]
        else: wOut,outNR,outNC=0,0,0
        
        # Embed + norm
        emb=w['token_embd.weight']
        if emb.shape[0]!=self.V: emb=emb.T
        self._emb=np.ascontiguousarray(emb)
        
        # Buffer arrays (keep Python refs)
        self._kvK=np.zeros((L,MP,NKH*HD),dtype=np.float32)
        self._kvV=np.zeros((L,MP,NKH*HD),dtype=np.float32)
        self._kvL=np.zeros(L,dtype=np.int32)
        self._x=np.zeros(N,dtype=np.float32); self._xn=np.zeros(N,dtype=np.float32)
        self._res=np.zeros(N,dtype=np.float32)
        NK=self.NH*HD
        self._q=np.zeros(NK,dtype=np.float32); self._k=np.zeros(NKH*HD,dtype=np.float32)
        self._v=np.zeros(NKH*HD,dtype=np.float32); self._att=np.zeros(NK,dtype=np.float32)
        self._gate=np.zeros(FF,dtype=np.float32); self._up=np.zeros(FF,dtype=np.float32)
        self._silu=np.zeros(FF,dtype=np.float32); self._o=np.zeros(N,dtype=np.float32)
        self._ffn=np.zeros(N,dtype=np.float32)
        self._lg=np.zeros(self.V,dtype=np.float32)
        
        def arr(a,t): return (t*len(a))(*a)
        
        # Build EC struct
        self.ec=EC()
        self.ec.L=L; self.ec.N=N; self.ec.NH=self.NH; self.ec.NKH=NKH
        self.ec.HD=HD; self.ec.FF=FF; self.ec.V=self.V; self.ec.eps=self.eps
        self.ec.nc=N
        self.ec.wQ=ctypes.cast(arr(wQ,ctypes.c_void_p),ctypes.c_void_p)
        self.ec.wK=ctypes.cast(arr(wK,ctypes.c_void_p),ctypes.c_void_p)
        self.ec.wV=ctypes.cast(arr(wV,ctypes.c_void_p),ctypes.c_void_p)
        self.ec.wO=ctypes.cast(arr(wO,ctypes.c_void_p),ctypes.c_void_p)
        self.ec.wG=ctypes.cast(arr(wG,ctypes.c_void_p),ctypes.c_void_p)
        self.ec.wU=ctypes.cast(arr(wU,ctypes.c_void_p),ctypes.c_void_p)
        self.ec.wD=ctypes.cast(arr(wD,ctypes.c_void_p),ctypes.c_void_p)
        self.ec.wAN=ctypes.cast(arr(wAN,ctypes.c_void_p),ctypes.c_void_p)
        self.ec.wFN=ctypes.cast(arr(wFN,ctypes.c_void_p),ctypes.c_void_p)
        self.ec.nQ=ctypes.cast(arr(nQ,ctypes.c_int),ctypes.c_void_p)
        self.ec.nK=ctypes.cast(arr(nK,ctypes.c_int),ctypes.c_void_p)
        self.ec.nV=ctypes.cast(arr(nV,ctypes.c_int),ctypes.c_void_p)
        self.ec.nO=ctypes.cast(arr(nO,ctypes.c_int),ctypes.c_void_p)
        self.ec.nG=ctypes.cast(arr(nG,ctypes.c_int),ctypes.c_void_p)
        self.ec.nU=ctypes.cast(arr(nU,ctypes.c_int),ctypes.c_void_p)
        self.ec.nD=ctypes.cast(arr(nD,ctypes.c_int),ctypes.c_void_p)
        self.ec.emb=self._emb.ctypes.data_as(ctypes.c_void_p)
        self.ec.onw=w['output_norm.weight'].ctypes.data_as(ctypes.c_void_p)
        self.ec.wOut=wOut; self.ec.outNR=outNR; self.ec.outNC=outNC
        self.ec.kvK=self._kvK.ctypes.data_as(ctypes.c_void_p)
        self.ec.kvV=self._kvV.ctypes.data_as(ctypes.c_void_p)
        self.ec.kvL=self._kvL.ctypes.data_as(ctypes.c_void_p)
        self.ec.MP=MP
        self.ec.x=self._x.ctypes.data_as(ctypes.c_void_p)
        self.ec.xn=self._xn.ctypes.data_as(ctypes.c_void_p)
        self.ec.res=self._res.ctypes.data_as(ctypes.c_void_p)
        self.ec.q=self._q.ctypes.data_as(ctypes.c_void_p)
        self.ec.k=self._k.ctypes.data_as(ctypes.c_void_p)
        self.ec.v=self._v.ctypes.data_as(ctypes.c_void_p)
        self.ec.att=self._att.ctypes.data_as(ctypes.c_void_p)
        self.ec.gate=self._gate.ctypes.data_as(ctypes.c_void_p)
        self.ec.up=self._up.ctypes.data_as(ctypes.c_void_p)
        self.ec.silu=self._silu.ctypes.data_as(ctypes.c_void_p)
        self.ec.oproj=self._o.ctypes.data_as(ctypes.c_void_p)
        self.ec.ffn=self._ffn.ctypes.data_as(ctypes.c_void_p)
    
    def reset(self): self._kvL[:]=0; self.pos=0
    
    def forward(self,tok):
        self.lib.engine_forward(ctypes.c_int(tok),ctypes.byref(self.ec),
                                self._lg.ctypes.data_as(ctypes.c_void_p))
        self.pos+=1
        return self._lg
