#!/usr/bin/env python3
"""CEngine2 — pure C forward pass via flat-array function. No struct bugs."""
import sys,os,ctypes,numpy as np,gguf
from gguf.constants import GGMLQuantizationType as QT
sys.path.insert(0,'/onedev-workspace/work/src/mojollama/kernels')
os.environ['PYTHONUNBUFFERED']='1'
kernel_dir='/onedev-workspace/work/src/mojollama/kernels'
BLS={2:18,3:20,12:144,13:176,14:210,8:34}

class CEngine2:
    def __init__(self, model_path, n_threads=32):
        os.environ['OMP_NUM_THREADS']=str(n_threads)
        self.lib=ctypes.CDLL(os.path.join(kernel_dir,'cengine2.so'))
        reader=gguf.GGUFReader(model_path)
        self._meta(reader)
        w,rw,wi=self._weights(reader)
        self._build(w,rw,wi)
        print(f"CEngine2: {self.L}L/{self.N}D/{self.FF}FF/{self.NH}H/{self.NKH}KV | t={n_threads}")
    
    def _meta(self,reader):
        f=reader.fields
        def g(n):
            for k,v in f.items():
                if k==n:
                    p=v.parts[-1] if hasattr(v,'parts') and len(v.parts)>=1 else None
                    if p is not None: return int(p[0]) if hasattr(p,'__iter__') and len(p)==1 else p
        a='llama'
        for p in ['qwen3moe','qwen2moe','llama','mistral']:
            if any(f'{p}.block_count' in k for k in f): a=p; break
        self.L=int(g(f'{a}.block_count') or 16)
        self.N=int(g(f'{a}.embedding_length') or 2048)
        self.FF=int(g(f'{a}.feed_forward_length') or self.N*4)
        self.NH=int(g(f'{a}.attention.head_count') or 32)
        self.NKH=int(g(f'{a}.attention.head_count_kv') or self.NH)
        self.HD=int(g(f'{a}.attention.key_length') or (self.N//self.NH))
        self.eps=float(g(f'{a}.attention.layer_norm_rms_epsilon') or 1e-6)
        self.V=int(g(f'{a}.vocab_size') or 0)
    
    def _weights(self,reader):
        w={};rw={};wi={};CQ={0,2,3,8,12,13,14}
        for t in reader.tensors:
            n=t.name;qt=QT(t.tensor_type).value
            if len(t.shape)==2:
                id_,od=int(t.shape[0]),int(t.shape[1])
                f32=gguf.dequantize(t.data,t.tensor_type).astype(np.float32).reshape(od,id_)
                w[n]=np.ascontiguousarray(f32)
                if qt in CQ:
                    r=np.ascontiguousarray(t.data.reshape(-1),dtype=np.uint8).copy()
                    rw[n]=r;wi[n]=(od,id_,BLS.get(qt,0),qt)
            elif len(t.shape)==1:
                f32=gguf.dequantize(t.data,t.tensor_type).astype(np.float32).reshape(-1)
                w[n]=np.ascontiguousarray(f32)
        e=w['token_embd.weight']
        if e.ndim==2 and e.shape[0]!=self.V: w['token_embd.weight']=np.ascontiguousarray(e.T)
        if not self.V: self.V=w['token_embd.weight'].shape[0]
        return w,rw,wi
    
    def _requant(self,rw,info):
        nr,nc=info[0],info[1];nb=nc//32
        omp=ctypes.CDLL(os.path.join(kernel_dir,'quant_kernels_omp.so'))
        cu=ctypes.POINTER(ctypes.c_uint8);cf=ctypes.POINTER(ctypes.c_float)
        omp.q6_k_dequantize_row.argtypes=[cu,cf,ctypes.c_int];omp.q6_k_dequantize_row.restype=None
        f32=np.zeros((nr,nc),dtype=np.float32)
        rp=rw.ctypes.data_as(cu);ba=ctypes.addressof(rp.contents)
        for r in range(nr):
            ptr=ctypes.cast(ba+r*(nc//256)*210,cu)
            omp.q6_k_dequantize_row(ptr,f32[r].ctypes.data_as(cf),ctypes.c_int(nc))
        blks=f32.reshape(nr,nb,32);amax=np.max(np.abs(blks),axis=2,keepdims=True)
        amax=np.clip(amax,1e-10,None);d=amax/127.0
        qs=np.clip(np.round(blks/d),-127,127).astype(np.int8)
        dh=d[:,:,0].astype(np.float16).view(np.uint16)
        q8=np.zeros((nr,nb,34),dtype=np.uint8)
        q8[:,:,0]=dh&0xFF;q8[:,:,1]=(dh>>8)&0xFF;q8[:,:,2:]=qs.view(np.uint8)
        self._ob=np.ascontiguousarray(q8.reshape(-1),dtype=np.uint8)
        print(f"  Output Q8_0: {nr}×{nc} ({len(self._ob)/1024/1024:.0f} MB)")
        return self._ob.ctypes.data_as(ctypes.c_void_p)
    
    def _build(self,w,rw,wi):
        N=self.N;NKH=self.NKH;HD=self.HD;FF=self.FF;L=self.L;MP=4096
        # Per-layer: store NUMPY arrays (not ctypes wrappers) to keep refs alive
        self._wQ=[];self._wK=[];self._wV=[];self._wO=[];self._wG=[];self._wU=[];self._wD=[]
        self._nQ=[];self._nK=[];self._nV=[];self._nO=[];self._nG=[];self._nU=[];self._nD=[]
        out_n='output.weight' if 'output.weight' in w else 'token_embd.weight'
        for i in range(L):
            pfx=f'blk.{i}'
            for n,arr,narr in [('attn_q','_wQ','_nQ'),('attn_k','_wK','_nK'),('attn_v','_wV','_nV'),
                              ('attn_output','_wO','_nO'),('ffn_gate','_wG','_nG'),
                              ('ffn_up','_wU','_nU'),('ffn_down','_wD','_nD')]:
                inf=wi.get(f'{pfx}.{n}.weight')
                if inf:
                    getattr(self,arr).append(rw[f'{pfx}.{n}.weight'])
                    getattr(self,narr).append(inf[0])
                else:
                    getattr(self,arr).append(np.zeros(1,dtype=np.uint8))
                    getattr(self,narr).append(0)
        # Norm weights
        self._wAN=[w[f'blk.{i}.attn_norm.weight'] for i in range(L)]
        self._wFN=[w[f'blk.{i}.ffn_norm.weight'] for i in range(L)]
        
        # Output
        oi=wi.get(out_n)
        if oi and oi[3] in (6,14,12,10):
            self._wOut=self._requant(rw[out_n],oi)
            self.outNR,self.outNC=oi[0],oi[1]
        elif oi:
            self._wOut=rw[out_n].ctypes.data_as(ctypes.c_void_p)
            self.outNR,self.outNC=oi[0],oi[1]
        else: self._wOut=0;self.outNR=0;self.outNC=0
        
        # Embed + norm
        e=w['token_embd.weight']
        if e.shape[0]!=self.V: e=e.T
        self._emb=np.ascontiguousarray(e)
        self._onw=w['output_norm.weight']
        
        # Buffers
        self._kvK=np.zeros((L,MP,NKH*HD),dtype=np.float32)
        self._kvV=np.zeros((L,MP,NKH*HD),dtype=np.float32)
        self._kvL=np.zeros(L,dtype=np.int32)
        NK=self.NH*HD
        self._x=np.zeros(N,dtype=np.float32);self._xn=np.zeros(N,dtype=np.float32)
        self._res=np.zeros(N,dtype=np.float32)
        self._q=np.zeros(NK,dtype=np.float32);self._k=np.zeros(NKH*HD,dtype=np.float32)
        self._v=np.zeros(NKH*HD,dtype=np.float32);self._att=np.zeros(NK,dtype=np.float32)
        self._gate=np.zeros(FF,dtype=np.float32);self._up=np.zeros(FF,dtype=np.float32)
        self._silu=np.zeros(FF,dtype=np.float32);self._o=np.zeros(N,dtype=np.float32)
        self._ffn=np.zeros(N,dtype=np.float32);self._lg=np.zeros(self.V,dtype=np.float32)
        
        # Keep ctypes pointer arrays alive as instance vars
        self._ct_wQ=self._aptr([a.ctypes.data_as(ctypes.c_void_p) for a in self._wQ],ctypes.c_void_p)
        self._ct_wK=self._aptr([a.ctypes.data_as(ctypes.c_void_p) for a in self._wK],ctypes.c_void_p)
        self._ct_wV=self._aptr([a.ctypes.data_as(ctypes.c_void_p) for a in self._wV],ctypes.c_void_p)
        self._ct_wO=self._aptr([a.ctypes.data_as(ctypes.c_void_p) for a in self._wO],ctypes.c_void_p)
        self._ct_wG=self._aptr([a.ctypes.data_as(ctypes.c_void_p) for a in self._wG],ctypes.c_void_p)
        self._ct_wU=self._aptr([a.ctypes.data_as(ctypes.c_void_p) for a in self._wU],ctypes.c_void_p)
        self._ct_wD=self._aptr([a.ctypes.data_as(ctypes.c_void_p) for a in self._wD],ctypes.c_void_p)
        self._ct_wAN=self._aptr([a.ctypes.data_as(ctypes.c_void_p) for a in self._wAN],ctypes.c_void_p)
        self._ct_wFN=self._aptr([a.ctypes.data_as(ctypes.c_void_p) for a in self._wFN],ctypes.c_void_p)
        self._ct_nQ=self._aptr([ctypes.c_int(a) for a in self._nQ],ctypes.c_int)
        self._ct_nK=self._aptr([ctypes.c_int(a) for a in self._nK],ctypes.c_int)
        self._ct_nV=self._aptr([ctypes.c_int(a) for a in self._nV],ctypes.c_int)
        self._ct_nO=self._aptr([ctypes.c_int(a) for a in self._nO],ctypes.c_int)
        self._ct_nG=self._aptr([ctypes.c_int(a) for a in self._nG],ctypes.c_int)
        self._ct_nU=self._aptr([ctypes.c_int(a) for a in self._nU],ctypes.c_int)
        self._ct_nD=self._aptr([ctypes.c_int(a) for a in self._nD],ctypes.c_int)
    
    def _aptr(self,arr,typ): return (typ*len(arr))(*arr)
    
    def _setup_fn(self):
        """Prepare the engine_forward ctypes call."""
        ci=ctypes.c_int;cv=ctypes.c_void_p;cf=ctypes.POINTER(ctypes.c_float)
        # engine_forward(tok, L, N, NH, NKH, HD, FF, V, eps,
        #   wQ, wK, wV, wO, wG, wU, wD, wAN, wFN,  (9 ptr arrays)
        #   nQ, nK, nV, nO, nG, nU, nD,             (7 int arrays)
        #   nc, emb, onw, wOut, outNR, outNC,        (misc)
        #   kvK, kvV, kvL, MP,                        (KV cache)
        #   x, xn, res, q, k, v, att, gate, up, silu, oproj, ffn, (12 bufs)
        #   logits)
        args = [
            ci, ci, ci, ci, ci, ci, ci, ci, ctypes.c_float,  # dims
            cv, cv, cv, cv, cv, cv, cv, cv, cv,  # weight arrays
            cv, cv, cv, cv, cv, cv, cv,  # row count arrays
            ci, cv, cv, cv, ci, ci,  # embed/out
            cv, cv, cv, ci,  # KV cache
            cv, cv, cv, cv, cv, cv, cv, cv, cv, cv, cv, cv,  # 12 buf ptrs
            cv  # logits
        ]
        self.lib.engine_forward.argtypes = args
        self.lib.engine_forward.restype = None
    
    def reset(self): self._kvL[:]=0
    
    def forward(self, tok):
        if not hasattr(self,'_fn_setup'):
            self._setup_fn(); self._fn_setup=True
        
        # Build call args
        bufs = [self._x, self._xn, self._res, self._q, self._k, self._v,
                self._att, self._gate, self._up, self._silu, self._o, self._ffn]
        
        self.lib.engine_forward(
            ci(tok), ci(self.L), ci(self.N), ci(self.NH), ci(self.NKH),
            ci(self.HD), ci(self.FF), ci(self.V), ctypes.c_float(self.eps),
            self._ct_wQ, self._ct_wK, self._ct_wV,
            self._ct_wO, self._ct_wG, self._ct_wU, self._ct_wD,
            self._ct_wAN, self._ct_wFN,
            self._ct_nQ, self._ct_nK, self._ct_nV,
            self._ct_nO, self._ct_nG, self._ct_nU, self._ct_nD,
            ci(self.N),
            self._emb.ctypes.data_as(ctypes.c_void_p),
            self._onw.ctypes.data_as(ctypes.c_void_p),
            self._wOut, ci(self.outNR), ci(self.outNC),
            self._kvK.ctypes.data_as(ctypes.c_void_p),
            self._kvV.ctypes.data_as(ctypes.c_void_p),
            self._kvL.ctypes.data_as(ctypes.c_void_p),
            ci(4096),
            *[b.ctypes.data_as(ctypes.c_void_p) for b in bufs],
            self._lg.ctypes.data_as(ctypes.c_void_p)
        )
        return self._lg

ci=ctypes.c_int
