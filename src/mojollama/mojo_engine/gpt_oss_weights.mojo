// Auto-generated GPT-OSS weight indices (436 files)
// n_embd=2880 n_layers=24
// n_experts=32 n_experts_per_tok=4

@extern("load_weight")
def _load_w(idx: Int64) abi("C") -> Int64: ...
@extern("free_weight")
def _free_w(p: Int64) abi("C") -> None: ...
@extern("weight_size")
def _w_size(idx: Int64) abi("C") -> Int64: ...

comptime EMB: Int64 = 0
comptime EMB_F32: Int64 = 1
comptime META: Int64 = 434
comptime OUT_NORM: Int64 = 435

comptime L0_attn_norm: Int64 = 2
comptime L0_down_e: Int64 = 3
comptime L0_down_info: Int64 = 4
comptime L0_ffn_norm: Int64 = 5
comptime L0_gate_e: Int64 = 6
comptime L0_gate_info: Int64 = 7
comptime L0_k_info: Int64 = 8
comptime L0_k_w: Int64 = 9
comptime L0_o_f32: Int64 = 10
comptime L0_o_info: Int64 = 11
comptime L0_o_w: Int64 = 12
comptime L0_q_info: Int64 = 13
comptime L0_q_w: Int64 = 14
comptime L0_router: Int64 = 15
comptime L0_up_e: Int64 = 16
comptime L0_up_info: Int64 = 17
comptime L0_v_info: Int64 = 18
comptime L0_v_w: Int64 = 19

comptime N_EXPERTS: Int64 = 32
comptime N_EXP_PER_TOK: Int64 = 4
comptime N_LAYERS: Int64 = 24
comptime N_EMBD: Int64 = 2880
comptime N_HEAD: Int64 = 64
comptime N_KV_HEAD: Int64 = 8
comptime HEAD_DIM: Int64 = 64
comptime VOCAB_SIZE: Int64 = 201088
comptime N_FF: Int64 = 2880
comptime LAYER_STRIDE: Int64 = 18

