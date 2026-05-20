"""ZAYA BC struct builder for zaya_batch_forward."""
import ctypes, numpy as np

class BC(ctypes.Structure):
    _fields_ = [
        ('L', ctypes.c_int), ('N', ctypes.c_int), ('NH', ctypes.c_int),
        ('NKH', ctypes.c_int), ('HD', ctypes.c_int), ('FF', ctypes.c_int),
        ('V', ctypes.c_int), ('eps', ctypes.c_float),
        ('wQ', ctypes.POINTER(ctypes.c_void_p)),
        ('wK', ctypes.POINTER(ctypes.c_void_p)),
        ('wV', ctypes.POINTER(ctypes.c_void_p)),
        ('wO', ctypes.POINTER(ctypes.c_void_p)),
        ('wG', ctypes.POINTER(ctypes.c_void_p)),
        ('wU', ctypes.POINTER(ctypes.c_void_p)),
        ('wD', ctypes.POINTER(ctypes.c_void_p)),
        ('wAN', ctypes.POINTER(ctypes.c_void_p)),
        ('wFN', ctypes.POINTER(ctypes.c_void_p)),
        ('nQ', ctypes.POINTER(ctypes.c_int)),
        ('nK', ctypes.POINTER(ctypes.c_int)),
        ('nV', ctypes.POINTER(ctypes.c_int)),
        ('nO', ctypes.POINTER(ctypes.c_int)),
        ('nG', ctypes.POINTER(ctypes.c_int)),
        ('nU', ctypes.POINTER(ctypes.c_int)),
        ('nD', ctypes.POINTER(ctypes.c_int)),
        ('nc', ctypes.c_int),
        ('emb', ctypes.POINTER(ctypes.c_float)),
        ('onw', ctypes.POINTER(ctypes.c_float)),
        ('wOut', ctypes.POINTER(ctypes.c_uint8)),
        ('outNR', ctypes.c_int), ('outNC', ctypes.c_int), ('outQuant', ctypes.c_int),
        ('kv_array', ctypes.POINTER(ctypes.c_void_p)),
        ('logits', ctypes.POINTER(ctypes.c_float)),
        ('n_experts', ctypes.c_int), ('n_experts_per_tok', ctypes.c_int),
        ('moe_intermediate', ctypes.c_int),
        ('w_gate_inp', ctypes.POINTER(ctypes.c_void_p)),
        ('w_gate_exps', ctypes.POINTER(ctypes.c_void_p)),
        ('w_up_exps', ctypes.POINTER(ctypes.c_void_p)),
        ('w_down_exps', ctypes.POINTER(ctypes.c_void_p)),
        ('gate_exp_quant', ctypes.c_int), ('up_exp_quant', ctypes.c_int),
        ('down_exp_quant', ctypes.c_int),
        ('q_quant', ctypes.POINTER(ctypes.c_int)),
        ('k_quant', ctypes.POINTER(ctypes.c_int)),
        ('v_quant', ctypes.POINTER(ctypes.c_int)),
        ('o_quant', ctypes.POINTER(ctypes.c_int)),
        ('g_quant', ctypes.POINTER(ctypes.c_int)),
        ('u_quant', ctypes.POINTER(ctypes.c_int)),
        ('d_quant', ctypes.POINTER(ctypes.c_int)),
        ('emb_quant', ctypes.c_int),
        ('wQK', ctypes.POINTER(ctypes.c_void_p)),
        ('qk_quant', ctypes.POINTER(ctypes.c_int)),
        ('cos_table', ctypes.POINTER(ctypes.c_float)),
        ('sin_table', ctypes.POINTER(ctypes.c_float)),
        ('max_ctx', ctypes.c_int),
        ('workspace', ctypes.POINTER(ctypes.c_float)),
        ('ws_size', ctypes.c_int),
        ('rope_dim', ctypes.c_int),
        ('full_attn_interval', ctypes.c_int),
        ('wQKV', ctypes.POINTER(ctypes.c_void_p)),
        ('qkv_quant', ctypes.POINTER(ctypes.c_int)),
        ('wAttnG', ctypes.POINTER(ctypes.c_void_p)),
        ('attnG_quant', ctypes.POINTER(ctypes.c_int)),
        ('ssm_conv1d', ctypes.POINTER(ctypes.c_void_p)),
        ('ssm_a', ctypes.POINTER(ctypes.c_void_p)),
        ('ssm_dt_bias', ctypes.POINTER(ctypes.c_void_p)),
        ('ssm_alpha', ctypes.POINTER(ctypes.c_void_p)),
        ('ssm_beta', ctypes.POINTER(ctypes.c_void_p)),
        ('ssm_norm', ctypes.POINTER(ctypes.c_void_p)),
        ('wSsmOut', ctypes.POINTER(ctypes.c_void_p)),
        ('ssm_out_quant', ctypes.POINTER(ctypes.c_int)),
        ('ssm_state', ctypes.POINTER(ctypes.c_float)),
        ('wSHexpG', ctypes.POINTER(ctypes.c_void_p)),
        ('wSHexpU', ctypes.POINTER(ctypes.c_void_p)),
        ('wSHexpD', ctypes.POINTER(ctypes.c_void_p)),
        ('shexp_g_quant', ctypes.POINTER(ctypes.c_int)),
        ('shexp_u_quant', ctypes.POINTER(ctypes.c_int)),
        ('shexp_d_quant', ctypes.POINTER(ctypes.c_int)),
        ('wShexpRouter', ctypes.POINTER(ctypes.c_void_p)),
        ('layer_types', ctypes.POINTER(ctypes.c_int)),
        ('n_layers_actual', ctypes.c_int),
        # ZAYA-specific
        ('w_zaya_ffn_gate_inp', ctypes.POINTER(ctypes.c_void_p)),
        ('w_zaya_ffn_gate', ctypes.POINTER(ctypes.c_void_p)),
        ('w_zaya_mlp2', ctypes.POINTER(ctypes.c_void_p)),
        ('w_zaya_mlp4', ctypes.POINTER(ctypes.c_void_p)),
        ('w_zaya_router_bias', ctypes.POINTER(ctypes.c_void_p)),
        ('w_zaya_ffn_norm', ctypes.POINTER(ctypes.c_void_p)),
        ('w_zaya_res_hs_w', ctypes.POINTER(ctypes.c_void_p)),
        ('w_zaya_res_hs_b', ctypes.POINTER(ctypes.c_void_p)),
        ('w_zaya_res_res_w', ctypes.POINTER(ctypes.c_void_p)),
        ('w_zaya_res_res_b', ctypes.POINTER(ctypes.c_void_p)),
        ('zaya_ffn_gate_inp_quant', ctypes.c_int),
        ('zaya_ffn_gate_quant', ctypes.c_int),
        ('zaya_mlp2_quant', ctypes.c_int),
        ('zaya_mlp4_quant', ctypes.c_int),
        ('has_moe_layer', ctypes.POINTER(ctypes.c_int)),
        ('zaya_expert_n', ctypes.c_int),
        ('zaya_moe_intermediate', ctypes.c_int),
        ('w_cca_v1', ctypes.POINTER(ctypes.c_void_p)),
        ('w_cca_v2', ctypes.POINTER(ctypes.c_void_p)),
        ('w_ssm_conv1d', ctypes.POINTER(ctypes.c_void_p)),
        ('cca_v1_quant', ctypes.c_int),
        ('cca_v2_quant', ctypes.c_int),
    ]

def build_zaya_bc(engine):
    """Build BC struct for ZAYA batch forward."""
    e = engine
    cu8 = ctypes.POINTER(ctypes.c_uint8)
    cf = ctypes.POINTER(ctypes.c_float)
    ci = ctypes.POINTER(ctypes.c_int)
    L = e.n_layers; N = e.n_embd
    
    def _pp(name):
        """Get ctypes pointer from raw_weights for a weight name."""
        rw = e.raw_weights.get(name)
        if rw is None: return ctypes.cast(0, cu8)
        return rw.ctypes.data_as(cu8)
    
    def _fp(name):
        """Get float pointer from weights."""
        w = e.weights.get(name)
        if w is None: return ctypes.cast(0, cf)
        return w.ctypes.data_as(cf)
    
    def _arr_pp(prefix, suffix, n):
        """Array of uint8 pointers for each layer."""
        arr = (cu8 * n)()
        for i in range(n):
            arr[i] = _pp(f'{prefix}{i}{suffix}')
        return ctypes.cast(arr, ctypes.POINTER(ctypes.c_void_p))
    
    def _arr_fp(prefix, suffix, n):
        """Array of float pointers for each layer."""
        arr = (cf * n)()
        for i in range(n):
            arr[i] = _fp(f'{prefix}{i}{suffix}')
        return ctypes.cast(arr, ctypes.POINTER(ctypes.c_void_p))
    
    def _arr_int(prefix, suffix, n, val_fn):
        """Array of ints for each layer."""
        arr = (ctypes.c_int * n)()
        for i in range(n):
            arr[i] = val_fn(i)
        return ctypes.cast(arr, ci)
    
    bc = BC()
    bc.L = ctypes.c_int(L)
    bc.N = ctypes.c_int(N)
    bc.NH = ctypes.c_int(e.n_head)
    bc.NKH = ctypes.c_int(e.n_kv_head)
    bc.HD = ctypes.c_int(e.head_dim)
    bc.FF = ctypes.c_int(getattr(e, 'n_ff', N))
    bc.V = ctypes.c_int(e.vocab_size)
    bc.eps = ctypes.c_float(1e-5)
    
    bc.wQ = _arr_pp('blk.', '.attn_q.weight', L)
    bc.wK = _arr_pp('blk.', '.attn_k.weight', L)
    bc.wV = ctypes.cast(0, ctypes.POINTER(ctypes.c_void_p))  # V = K (no separate V)
    bc.wO = _arr_pp('blk.', '.attn_output.weight', L)
    bc.wG = ctypes.cast(0, ctypes.POINTER(ctypes.c_void_p))  # no separate gate
    bc.wU = ctypes.cast(0, ctypes.POINTER(ctypes.c_void_p))
    bc.wD = ctypes.cast(0, ctypes.POINTER(ctypes.c_void_p))
    bc.wAN = _arr_fp('blk.', '.attn_norm.weight', L)
    bc.wFN = _arr_fp('blk.', '.attn_norm.weight', L)  # same as attn_norm for ZAYA
    
    wi = e.weight_info
    def qv(n): info = wi.get(n); return info[3] if info else 0
    def nr(n): info = wi.get(n); return info[0] if info else 0
    def nc(n): info = wi.get(n); return info[1] if info else 0
    
    bc.nQ = _arr_int('blk.', '.attn_q.weight', L, lambda i: nr(f'blk.{i}.attn_q.weight'))
    bc.nK = _arr_int('blk.', '.attn_k.weight', L, lambda i: nr(f'blk.{i}.attn_k.weight'))
    bc.nV = _arr_int('', '', L, lambda i: 0)
    bc.nO = _arr_int('blk.', '.attn_output.weight', L, lambda i: nr(f'blk.{i}.attn_output.weight'))
    bc.nG = _arr_int('', '', L, lambda i: 0)
    bc.nU = _arr_int('', '', L, lambda i: 0)
    bc.nD = _arr_int('', '', L, lambda i: 0)
    
    bc.q_quant = _arr_int('blk.', '.attn_q.weight', L, lambda i: qv(f'blk.{i}.attn_q.weight'))
    bc.k_quant = _arr_int('blk.', '.attn_k.weight', L, lambda i: qv(f'blk.{i}.attn_k.weight'))
    bc.v_quant = _arr_int('', '', L, lambda i: 8)
    bc.o_quant = _arr_int('blk.', '.attn_output.weight', L, lambda i: qv(f'blk.{i}.attn_output.weight'))
    bc.g_quant = _arr_int('', '', L, lambda i: 8)
    bc.u_quant = _arr_int('', '', L, lambda i: 8)
    bc.d_quant = _arr_int('', '', L, lambda i: 8)
    bc.nc = ctypes.c_int(N)
    bc.emb = e.emb.ctypes.data_as(cf)
    bc.onw = _fp('output_norm.weight')
    
    # Output weight (embedding for tied ZAYA)
    out_name = 'token_embd.weight'
    out_info = wi.get(out_name)
    if out_info:
        bc.wOut = _pp(out_name)
        bc.outNR = ctypes.c_int(out_info[0])
        bc.outNC = ctypes.c_int(out_info[1])
        bc.outQuant = ctypes.c_int(out_info[3])
        bc.emb_quant = ctypes.c_int(out_info[3])
    
    bc.n_experts = ctypes.c_int(e.n_experts or 16)
    bc.n_experts_per_tok = ctypes.c_int(1)
    bc.moe_intermediate = ctypes.c_int(4096)  # n_ff for ZAYA
    
    bc.cos_table = ctypes.cast(0, cf)  # not used for ZAYA C path
    bc.sin_table = ctypes.cast(0, cf)
    bc.max_ctx = ctypes.c_int(4096)
    bc.rope_dim = ctypes.c_int(64)
    
    # MoE expert weights
    bc.w_gate_inp = _arr_fp('blk.', '.ffn_gate_inp.weight', L)
    bc.w_gate_exps = _arr_pp('blk.', '.ffn_gate_up_exps.weight', L)
    bc.w_up_exps = _arr_pp('blk.', '.ffn_gate_up_exps.weight', L)  # same fused
    bc.w_down_exps = _arr_pp('blk.', '.ffn_down_exps.weight', L)
    bc.gate_exp_quant = ctypes.c_int(qv('blk.1.ffn_gate_up_exps.weight'))
    bc.up_exp_quant = ctypes.c_int(qv('blk.1.ffn_gate_up_exps.weight'))
    bc.down_exp_quant = ctypes.c_int(qv('blk.1.ffn_down_exps.weight'))
    
    # ZAYA-specific
    bc.w_zaya_ffn_gate_inp = _arr_pp('blk.', '.ffn_gate_inp.weight', L)
    bc.w_zaya_ffn_gate = _arr_pp('blk.', '.ffn_gate.weight', L)
    bc.w_zaya_mlp2 = _arr_pp('blk.', '.zaya_router_mlp2.weight', L)
    bc.w_zaya_mlp4 = _arr_pp('blk.', '.zaya_router_mlp4.weight', L)
    bc.w_zaya_router_bias = _arr_fp('blk.', '.zaya_router_biases.weight', L)
    bc.w_zaya_ffn_norm = _arr_fp('blk.', '.attn_norm.weight', L)
    
    bc.w_zaya_res_hs_w = _arr_fp('blk.', '.res_scale_hs.weight', L)
    bc.w_zaya_res_hs_b = _arr_fp('blk.', '.res_scale_hs.bias', L)
    bc.w_zaya_res_res_w = _arr_fp('blk.', '.res_scale_res.weight', L)
    bc.w_zaya_res_res_b = _arr_fp('blk.', '.res_scale_res.bias', L)
    
    bc.zaya_ffn_gate_inp_quant = ctypes.c_int(qv('blk.1.ffn_gate_inp.weight'))
    bc.zaya_ffn_gate_quant = ctypes.c_int(qv('blk.1.ffn_gate.weight'))
    bc.zaya_mlp2_quant = ctypes.c_int(qv('blk.1.zaya_router_mlp2.weight'))
    bc.zaya_mlp4_quant = ctypes.c_int(qv('blk.1.zaya_router_mlp4.weight'))
    
    has_moe = (ctypes.c_int * L)()
    for i in range(L): has_moe[i] = ctypes.c_int(1 if i % 2 else 0)
    bc.has_moe_layer = ctypes.cast(has_moe, ctypes.POINTER(ctypes.c_int))
    bc.zaya_expert_n = ctypes.c_int(e.n_experts or 16)
    bc.zaya_moe_intermediate = ctypes.c_int(4096)
    
    return bc

def run_zaya_batch_forward(engine, token_id):
    """Run ZAYA batch forward via C engine."""
    import time
    ce = getattr(engine, '_cengine', None)
    if ce is None:
        return None  # not available
    
    if not hasattr(ce, '_zaya_bc'):
        bc = build_zaya_bc(engine)
        ce._zaya_bc = bc
        
        # Workspace: 50 * S floats where S = max(N, nq, nk, 8192)
        S = max(engine.n_embd, engine.n_head * engine.head_dim, engine.n_kv_head * engine.head_dim, 8192)
        ws_size = 44 * S
        ce._zaya_ws = np.zeros(ws_size, dtype=np.float32)
    
    bc = ce._zaya_bc
    ws = ce._zaya_ws
    logits_out = engine._logits
    
    ce.zaya_batch_forward(
        ctypes.byref(bc),
        ctypes.byref(ctypes.c_int(token_id)),
        ctypes.c_int(1),
        ws.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    
    return engine._logits.copy()
