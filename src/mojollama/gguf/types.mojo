alias GGUF_MAGIC = 0x46554747
alias GGUF_VERSION = 3
alias GGUF_DEFAULT_ALIGNMENT: UInt64 = 32

struct GGUFValueType:
    alias UINT8 = 0
    alias INT8 = 1
    alias UINT16 = 2
    alias INT16 = 3
    alias UINT32 = 4
    alias INT32 = 5
    alias FLOAT32 = 6
    alias BOOL = 7
    alias STRING = 8
    alias ARRAY = 9
    alias UINT64 = 10
    alias INT64 = 11
    alias FLOAT64 = 12

struct GGMLType:
    alias F32 = 0
    alias F16 = 1
    alias Q4_0 = 2
    alias Q4_1 = 3
    alias Q8_0 = 7
    alias Q5_0 = 8
    alias Q5_1 = 9
    alias Q8_1 = 10
    alias Q2_K = 16
    alias Q3_K = 17
    alias Q4_K = 18
    alias Q5_K = 19
    alias Q6_K = 20
    alias Q8_K = 21
    alias IQ1_S = 26
    alias IQ1_M = 27
    alias IQ2_XXS = 28
    alias IQ2_XS = 29
    alias IQ2_S = 30
    alias IQ2_M = 33
    alias IQ3_XXS = 34
    alias IQ3_S = 35
    alias IQ3_M = 36
    alias IQ4_NL = 37
    alias IQ4_XS = 38

fn ggml_type_size(t: UInt32) -> UInt64:
    if t == GGMLType.F32:
        return 4
    elif t == GGMLType.F16:
        return 2
    elif t == GGMLType.Q4_0:
        return 18
    elif t == GGMLType.Q4_1:
        return 20
    elif t == GGMLType.Q8_0:
        return 26
    elif t == GGMLType.Q5_0:
        return 22
    elif t == GGMLType.Q5_1:
        return 24
    elif t == GGMLType.Q8_1:
        return 26
    elif t == GGMLType.Q2_K:
        return 36
    elif t == GGMLType.Q3_K:
        return 44
    elif t == GGMLType.Q4_K:
        return 72
    elif t == GGMLType.Q5_K:
        return 88
    elif t == GGMLType.Q6_K:
        return 52
    elif t == GGMLType.Q8_K:
        return 72
    elif t == GGMLType.IQ1_S:
        return 18
    elif t == GGMLType.IQ1_M:
        return 22
    elif t == GGMLType.IQ2_XXS:
        return 18
    elif t == GGMLType.IQ2_XS:
        return 18
    elif t == GGMLType.IQ2_S:
        return 22
    elif t == GGMLType.IQ2_M:
        return 26
    elif t == GGMLType.IQ3_XXS:
        return 18
    elif t == GGMLType.IQ3_S:
        return 22
    elif t == GGMLType.IQ3_M:
        return 26
    elif t == GGMLType.IQ4_NL:
        return 18
    elif t == GGMLType.IQ4_XS:
        return 18
    else:
        return 0

fn ggml_block_size(t: UInt32) -> UInt64:
    if t == GGMLType.F32:
        return 1
    elif t == GGMLType.F16:
        return 1
    elif t == GGMLType.Q4_0:
        return 32
    elif t == GGMLType.Q4_1:
        return 32
    elif t == GGMLType.Q8_0:
        return 32
    elif t == GGMLType.Q5_0:
        return 32
    elif t == GGMLType.Q5_1:
        return 32
    elif t == GGMLType.Q8_1:
        return 32
    elif t == GGMLType.Q2_K:
        return 256
    elif t == GGMLType.Q3_K:
        return 256
    elif t == GGMLType.Q4_K:
        return 256
    elif t == GGMLType.Q5_K:
        return 256
    elif t == GGMLType.Q6_K:
        return 256
    elif t == GGMLType.Q8_K:
        return 256
    elif t == GGMLType.IQ1_S:
        return 256
    elif t == GGMLType.IQ1_M:
        return 256
    elif t == GGMLType.IQ2_XXS:
        return 256
    elif t == GGMLType.IQ2_XS:
        return 256
    elif t == GGMLType.IQ2_S:
        return 256
    elif t == GGMLType.IQ2_M:
        return 256
    elif t == GGMLType.IQ3_XXS:
        return 256
    elif t == GGMLType.IQ3_S:
        return 256
    elif t == GGMLType.IQ3_M:
        return 256
    elif t == GGMLType.IQ4_NL:
        return 256
    elif t == GGMLType.IQ4_XS:
        return 256
    else:
        return 0

fn ggml_type_size_in_bytes(dtype: UInt32, n_elements: UInt64) -> UInt64:
    return n_elements * ggml_type_size(dtype) // ggml_block_size(dtype)
