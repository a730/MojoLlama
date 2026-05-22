# safetensors_extract.mojo — Pure Mojo safetensors → .bin extractor
# WHAT:  Wraps the C safetensors_extract helper. Reads a .safetensors file,
#        extracts all tensors to raw .bin weight files.
# WHY:   JSON parsing impossible in Mojo 1.0.0b1 (no String, no dicts).
#        C for I/O, Mojo orchestrates. Acceptable per "C for I/O" rule.
# WHEN:  2026-05-22
#
# Naming: safetensors → flat .bin files (dots→underscores, "model." stripped)
#   HF name                        → .bin file
#   lm_head.weight                 → lm_head_weight.bin
#   model.embed_tokens.weight      → embed_tokens_weight.bin
#   model.layers.N.self_attn.q     → layers_N_self_attn_q_proj_weight.bin
#   model.layers.N.mlp.gate_proj   → layers_N_mlp_gate_proj_weight.bin
#   model.norm.weight              → norm_weight.bin
#
# Usage: mojo safetensors_extract.mojo path/to/model.safetensors [out_dir]
#   (or build: mojo build safetensors_extract.mojo -o safetensors_extract_mojo)

from sys import args, exit

@extern("system")
def _c_system(cmd: UnsafePointer[UInt8, MutExternalOrigin]) -> Int: ...

def main():
    var argv = args()
    if argv.size < 2:
        print("Usage: safetensors_extract model.safetensors [out_dir]")
        print("Extracts all tensors from a .safetensors file to raw .bin files.")
        print("Default out_dir: ./weights/")
        print()
        print("Example: mojo safetensors_extract.mojo model.safetensors /tmp/weights/")
        exit(1)
    
    var inpath = argv[1]
    var outdir = argv[2] if argv.size > 2 else String("./weights")
    
    # The C binary must be compiled first: gcc -O2 safetensors_extract.c -o safetensors_extract
    # Located in the same directory as this Mojo file
    var binary = String("/onedev-workspace/work/src/mojollama/mojo_engine/safetensors_extract")
    
    # Build command: binary inpath outdir
    # Use a raw buffer to avoid String UInt8 bug
    var args_str = String(" ") + inpath + String(" ") + outdir
    var full_cmd = binary + args_str
    
    print("Extracting:", inpath, "→", outdir)
    var ret = _c_system(full_cmd.unsafe_ptr())
    
    if ret != 0:
        print("ERROR: safetensors_extract failed with code", ret)
        exit(ret)
