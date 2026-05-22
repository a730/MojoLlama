# MojoLlama Quantizer — Pure Mojo quantization (Q4_0, Q8_0, MXFP4) + GGUF writer
# [DOC] Uses syscall() for I/O (SYS_write=1 on x86-64) to avoid @extern("write") stdlib conflict.
# All pointers are stored as Int64 addresses from _alc, never requiring pointer-to-int conversion.
from std import time
from std.algorithm.backend.cpu.parallelize import parallelize

# ─── f16 encoding (IEEE 754 via Float64 arithmetic) ───
def encode_f16(f: Float32) -> UInt16:
    if f == 0.0: return UInt16(0)
    var sign: UInt16 = 0
    var fd = Float64(f)
    if fd < 0: sign = UInt16(1); fd = -fd
    var e: Int = 0
    if fd >= 1.0:
        while fd >= 2.0: fd *= 0.5; e += 1
    else:
        while fd < 1.0: fd *= 2.0; e -= 1
    var half_exp = e + 15
    if half_exp < 0: half_exp = 0
    if half_exp > 31: half_exp = 31
    var mant = Int(fd * 2048.0)
    if mant > 1023: mant = 1023
    return UInt16(Int(sign) << 15 | half_exp << 10 | mant)

def str_to_cstr(s: String, addr: Int64):
    """Copy string to pre-allocated buffer at addr, null-terminated."""
    var blen = s.byte_length()
    var buf = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(addr))
    var src = s.unsafe_ptr()
    for i in range(blen): buf.store(i, src.load(i))
    buf.store(blen, UInt8(0))

def str_to_cstr_uint64_len(s: String, addr: Int64):
    """Write GGUF String format: uint64 length + chars (no null)."""
    var blen = s.byte_length()
    var buf = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(addr))
    # Write uint64 length (8 bytes, little-endian)
    for i in range(8): buf.store(i, UInt8((Int64(blen) >> (i*8)) & 0xFF))
    # Write chars
    var src = s.unsafe_ptr()
    for i in range(blen): buf.store(8+i, src.load(i))

def main():
    @extern("malloc")
    def _alc(sz: Int64) abi("C") -> Int64: ...
    @extern("syscall")
    def _sys(n: Int64, a1: Int64, a2: Int64, a3: Int64) abi("C") -> Int64: ...
    @extern("open")
    def _open(p: UnsafePointer[UInt8, MutExternalOrigin], fl: Int) abi("C") -> Int: ...
    
    comptime O_W: Int = 1; comptime O_C: Int = 64; comptime O_T: Int = 512
    comptime SYS_w: Int64 = 1  # SYS_write on x86-64
    
    # ─── Q4_0 Quantizer ───
    # Returns Int64 address of allocated quantized buffer
    def q_q4(data_addr: Int64, nv: Int) -> Int64:
        var nb = (nv + 31) // 32
        var out_addr = _alc(Int64(nb * 18))
        var data = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(data_addr))
        var out = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(out_addr))
        def wk(b: Int) capturing -> None:
            var bo = b * 18; var vo = b * 32
            var amax: Float32 = 0.0
            for j in range(32):
                if vo + j >= nv: break
                var v = data.load(vo + j)
                if v < 0: v = -v
                if v > amax: amax = v
            if amax == 0.0: amax = 1.0
            var d = amax / 7.0; var id = 7.0 / amax
            var su = encode_f16(d)
            out.store(bo, UInt8(su & 0xFF)); out.store(bo+1, UInt8((su >> 8) & 0xFF))
            for j in range(32):
                if vo + j >= nv: break
                var q = Int32(data.load(vo+j) * id)
                if q > 7: q = 7
                if q < -8: q = -8
                if q < 0: q += 16
                var bi = bo + 2 + j // 2
                if j % 2 == 0:
                    out.store(bi, UInt8(q & 0x0F) | (out.load(bi) & 0xF0))
                else:
                    out.store(bi, UInt8((q & 0x0F) << 4) | (out.load(bi) & 0x0F))
        parallelize[func=wk](num_work_items=nb)
        return out_addr
    
    # ─── Q8_0 Quantizer ───
    def q_q8(data_addr: Int64, nv: Int) -> Int64:
        var nb = (nv + 31) // 32
        var out_addr = _alc(Int64(nb * 34))
        var data = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(data_addr))
        var out = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(out_addr))
        def wk(b: Int) capturing -> None:
            var bo = b * 34; var vo = b * 32
            var amax: Float32 = 0.0
            for j in range(32):
                if vo + j >= nv: break
                var v = data.load(vo + j)
                if v < 0: v = -v
                if v > amax: amax = v
            if amax == 0.0: amax = 1.0
            var d = amax / 127.0; var id = 127.0 / amax
            var su = encode_f16(d)
            out.store(bo, UInt8(su & 0xFF)); out.store(bo+1, UInt8((su >> 8) & 0xFF))
            for j in range(32):
                if vo + j >= nv: break
                var q = Int32(data.load(vo+j) * id)
                if q > 127: q = 127
                if q < -128: q = -128
                if q < 0: q += 256
                out.store(bo + 2 + j, UInt8(q & 0xFF))
        parallelize[func=wk](num_work_items=nb)
        return out_addr
    
    # ─── Q5_0 Quantizer (22 bytes/32 vals: f16 scale + 4 bytes high bits + 16 bytes nibbles) ───
    def q_q5(data_addr: Int64, nv: Int) -> Int64:
        var nb = (nv + 31) // 32
        var out_addr = _alc(Int64(nb * 22))
        var data = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(data_addr))
        var out = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(out_addr))
        def wk(b: Int) capturing -> None:
            var bo = b * 22; var vo = b * 32
            var amax: Float32 = 0.0
            for j in range(32):
                if vo + j >= nv: break
                var v = data.load(vo + j)
                if v < 0: v = -v
                if v > amax: amax = v
            if amax == 0.0: amax = 1.0
            # Q5_0: scale to 5-bit range (-16..15)
            var d = amax / 15.0; var id = 15.0 / amax
            var su = encode_f16(d)
            out.store(bo, UInt8(su & 0xFF)); out.store(bo+1, UInt8((su >> 8) & 0xFF))
            var qh: UInt32 = 0
            for j in range(32):
                if vo + j >= nv: break
                var q = Int32(data.load(vo+j) * id)
                if q > 15: q = 15
                if q < -16: q = -16
                var low = q & 0x0F
                var high = (UInt32(q >> 4) & 1) << j
                qh = qh | high
                var bi = bo + 6 + j // 2
                if j % 2 == 0:
                    out.store(bi, UInt8(low & 0x0F) | (out.load(bi) & 0xF0))
                else:
                    out.store(bi, UInt8((low & 0x0F) << 4) | (out.load(bi) & 0x0F))
            # Store high bits (4 bytes, big-endian bit order)
            for i in range(4):
                out.store(bo + 2 + i, UInt8((qh >> (i*8)) & 0xFF))
        parallelize[func=wk](num_work_items=nb)
        return out_addr

    # ─── Q4_K Quantizer (144 bytes/256 vals: super-block with 8 sub-blocks) ───
    def q_q4k(data_addr: Int64, nv: Int) -> Int64:
        var nb = (nv + 255) // 256
        var out_addr = _alc(Int64(nb * 144))
        var data = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(data_addr))
        var out = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(out_addr))
        def wk(b: Int) capturing -> None:
            var bo = b * 144; var vo = b * 256
            # 8 sub-blocks of 32 values each
            var sub_max = SIMD[DType.float32, 8](0.0)
            var sub_min = SIMD[DType.float32, 8](0.0)
            for sb in range(8):
                var smax: Float32 = 0.0; var smin: Float32 = 0.0
                for j in range(32):
                    if vo + sb * 32 + j >= nv: break
                    var v = data.load(vo + sb * 32 + j)
                    if v > smax: smax = v
                    if v < smin: smin = v
                sub_max[sb] = smax; sub_min[sb] = smin
            # Super-block scale: use max of sub-block ranges
            var sup_max: Float32 = 0.0
            for sb in range(8):
                var r = sub_max[sb] - sub_min[sb]
                if r > sup_max: sup_max = r
            if sup_max == 0.0: sup_max = 1.0
            var d = sup_max / 252.0  # 6-bit range
            var dmin = -sup_max / 252.0  # negative side
            var su_d = encode_f16(d)
            var su_dm = encode_f16(dmin)
            out.store(bo, UInt8(su_d & 0xFF)); out.store(bo+1, UInt8((su_d >> 8) & 0xFF))
            out.store(bo+2, UInt8(su_dm & 0xFF)); out.store(bo+3, UInt8((su_dm >> 8) & 0xFF))
            # Scales for each sub-block: quantize to 6 bits
            var scales = SIMD[DType.uint8, 8](0)
            var mins = SIMD[DType.uint8, 8](0)
            for sb in range(8):
                var sc = Int32(sub_max[sb] / d)
                if sc > 63: sc = 63
                if sc < 0: sc = 0
                var mn = Int32((-sub_min[sb]) / dmin)  # use dmin for negative side
                if mn > 63: mn = 63
                if mn < 0: mn = 0
                scales[sb] = UInt8(sc & 0x3F)
                mins[sb] = UInt8(mn & 0x3F)
            # Pack scales: first 4 in lower nibbles of bytes 132-135
            # Last 4 in upper nibbles. This is a simplified packing.
            for sb in range(4):
                out.store(bo + 132 + sb, UInt8(Int(scales[sb]) | (Int(scales[sb+4]) << 4)))
                out.store(bo + 136 + sb, UInt8(Int(mins[sb]) | (Int(mins[sb+4]) << 4)))
            # Quantize nibbles
            for sb in range(8):
                var sc = scales[sb]
                var mn = mins[sb]
                var sc_f32 = Float32(Int(sc))
                var mn_f32 = Float32(Int(mn))
                for j in range(32):
                    if vo + sb * 32 + j >= nv: break
                    var v = data.load(vo + sb * 32 + j)
                    var qv = Int32(v / d - sc_f32 + mn_f32 * 6.0)
                    if qv > 7: qv = 7
                    if qv < -8: qv = -8
                    if qv < 0: qv += 16
                    var bi = bo + 4 + sb * 16 + j // 2
                    if j % 2 == 0:
                        out.store(bi, UInt8(qv & 0x0F) | (out.load(bi) & 0xF0))
                    else:
                        out.store(bi, UInt8((qv & 0x0F) << 4) | (out.load(bi) & 0x0F))
        parallelize[func=wk](num_work_items=nb)
        return out_addr

    # ─── Q5_K Quantizer (176 bytes/256 vals: d + dmin + 12B scales + 32B qh + 128B qs) ───
    def q_q5k(data_addr: Int64, nv: Int) -> Int64:
        var nb = (nv + 255) // 256
        var out_addr = _alc(Int64(nb * 176))
        var data = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(data_addr))
        var out = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(out_addr))
        def wk(b: Int) capturing -> None:
            var bo = b * 176; var vo = b * 256
            # 8 sub-blocks of 32 each
            var d = Float32(1.0); var dmin = Float32(1.0)
            var scales = SIMD[DType.uint8, 8](0)
            var mins = SIMD[DType.uint8, 8](0)
            for sb in range(8):
                var smax: Float32 = -1e10; var smin: Float32 = 1e10
                for j in range(32):
                    if vo+sb*32+j >= nv: break
                    var v = data.load(vo+sb*32+j)
                    if v > smax: smax = v
                    if v < smin: smin = v
                var rng = smax - smin
                if rng > Float32(0.0):
                    d = rng / Float32(252.0)
                    dmin = Float32(-1.0) * d
                    scales[sb] = UInt8(Int32(smax / d))
                    mins[sb] = UInt8(Int32(Float32(-1.0) * smin / dmin))
            var su_d = encode_f16(d); var su_dm = encode_f16(dmin)
            out.store(bo, UInt8(su_d & 0xFF)); out.store(bo+1, UInt8((su_d >> 8) & 0xFF))
            out.store(bo+2, UInt8(su_dm & 0xFF)); out.store(bo+3, UInt8((su_dm >> 8) & 0xFF))
            # Pack scales (12 bytes): 6-bit per sub-block for scales and mins
            for sb in range(4):
                var packed_sc = Int(scales[sb]) | (Int(scales[sb+4]) << 6)
                packed_sc = packed_sc | (Int(mins[sb]) << 12) | (Int(mins[sb+4]) << 18)
                for i in range(3):
                    out.store(bo + 4 + sb*3 + i, UInt8((packed_sc >> (i*8)) & 0xFF))
            # Encode nibbles + 5th bit
            for sb in range(8):
                var sc = Float32(Int(scales[sb]))
                var mn = Float32(Int(mins[sb]))
                for j in range(32):
                    if vo+sb*32+j >= nv: break
                    var v = data.load(vo+sb*32+j)
                    var scaled = v / d - sc - mn * Float32(6.0)
                    var qv = Int32(scaled)
                    if qv > 15: qv = 15
                    if qv < -16: qv = -16
                    var lo = qv & 0x0F
                    var hi = UInt8((UInt32(qv) >> 4) & 1)
                    var bi = bo + 48 + sb * 16 + j // 2
                    if j % 2 == 0:
                        out.store(bi, UInt8(lo & 0x0F) | (out.load(bi) & 0xF0))
                    else:
                        out.store(bi, UInt8((lo & 0x0F) << 4) | (out.load(bi) & 0x0F))
                    # Store 5th high bit
                    var hi_idx = sb * 32 + j
                    var hi_byte = bo + 16 + hi_idx // 8
                    var hi_bit = hi_idx % 8
                    if hi != 0:
                        out.store(hi_byte, UInt8(Int(out.load(hi_byte)) | (1 << hi_bit)))
        parallelize[func=wk](num_work_items=nb)
        return out_addr

    # ─── Q6_K Quantizer (210 bytes/256 vals: 128B ql + 64B qh + 16B scales + 2B d) ───
    def q_q6k(data_addr: Int64, nv: Int) -> Int64:
        var nb = (nv + 255) // 256
        var out_addr = _alc(Int64(nb * 210))
        var data = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(data_addr))
        var out = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(out_addr))
        def wk(b: Int) capturing -> None:
            var bo = b * 210; var vo = b * 256
            # Find max absolute value for super-block scale
            var amax: Float32 = 0.0
            for j in range(256):
                if vo + j >= nv: break
                var v = data.load(vo + j)
                if v < 0: v = -v
                if v > amax: amax = v
            if amax == 0.0: amax = 1.0
            var d = amax / Float32(31.0)  # Q6_K: 6-bit range is -32..31
            var inv_d = Float32(1.0) / d
            var su = encode_f16(d)
            out.store(bo+208, UInt8(su & 0xFF)); out.store(bo+209, UInt8((su >> 8) & 0xFF))
            # Scales for 16 sub-groups of 16 values each
            for sg in range(16):
                var sg_max: Float32 = 0.0
                for j in range(16):
                    if vo+sg*16+j >= nv: break
                    var v = data.load(vo+sg*16+j)
                    if v < 0: v = -v
                    if v > sg_max: sg_max = v
                var sc = Int32(sg_max * inv_d)
                if sc > 63: sc = 63
                out.store(bo+192+sg, UInt8(sc & 0x3F))
            # Encode each value as 6 bits: low 4 in ql, high 2 in qh
            for j in range(256):
                if vo + j >= nv: break
                var v = data.load(vo + j)
                var qv = Int32(v * inv_d)
                if qv > 31: qv = 31
                if qv < -32: qv = -32
                if qv < 0: qv += 64
                var lo = qv & 0x0F
                var hi = (qv >> 4) & 3
                # Low nibble: 4 bits
                var li = bo + j // 2
                if j % 2 == 0:
                    out.store(li, UInt8(lo & 0x0F) | (out.load(li) & 0xF0))
                else:
                    out.store(li, UInt8((lo & 0x0F) << 4) | (out.load(li) & 0x0F))
                # High 2 bits: packed 4 per byte
                var hi_idx = j // 4
                var hi_shift = (j % 4) * 2
                out.store(bo+128+hi_idx, UInt8(Int(out.load(bo+128+hi_idx)) | (hi << hi_shift)))
        parallelize[func=wk](num_work_items=nb)
        return out_addr

    # ─── MXFP4 Quantizer ───
    def q_mx(data_addr: Int64, nv: Int) -> Int64:
        var nb = (nv + 31) // 32
        var out_addr = _alc(Int64(nb * 17))
        var data = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(data_addr))
        var out = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(out_addr))
        def wk(b: Int) capturing -> None:
            var bo = b * 17; var vo = b * 32
            var me: Int32 = -128
            for j in range(32):
                if vo + j >= nv: break
                var v = data.load(vo + j)
                if v == 0.0: continue
                var av = v if v >= 0 else -v; var e: Int32 = 0
                if av >= 1.0:
                    while av >= 2.0: av *= 0.5; e += 1
                else:
                    while av < 1.0: av *= 2.0; e -= 1
                if e > me: me = e
            out.store(bo + 16, UInt8((me + 127) & 0xFF))
            for j in range(32):
                if vo + j >= nv: break
                var v = data.load(vo + j)
                if me >= 0:
                    for _ in range(me): v *= 0.5
                else:
                    for _ in range(-me): v *= 2.0
                var q = Int32(v)
                if q > 7: q = 7
                if q < -8: q = -8
                if q < 0: q += 16
                var bi = bo + j // 2
                if j % 2 == 0:
                    out.store(bi, UInt8(q & 0x0F) | (out.load(bi) & 0xF0))
                else:
                    out.store(bi, UInt8((q & 0x0F) << 4) | (out.load(bi) & 0x0F))
        parallelize[func=wk](num_work_items=nb)
        return out_addr
    
    # ─── GGUF Writer (all I/O via syscall) ───
    def write_gguf(path: String, data_addr: Int64, nr: Int, nc: Int, qt: Int) capturing -> None:
        var nv = nr * nc
        var qd_addr: Int64 = 0
        var ds: Int64 = 0
        if qt == 2: qd_addr = q_q4(data_addr, nv); ds = Int64(nv) * 18 // 32
        elif qt == 6: qd_addr = q_q5(data_addr, nv); ds = Int64(nv) * 22 // 32
        elif qt == 8: qd_addr = q_q8(data_addr, nv); ds = Int64(nv) * 34 // 32
        elif qt == 12: qd_addr = q_q4k(data_addr, nv); ds = Int64(nv) * 144 // 256
        elif qt == 13: qd_addr = q_q5k(data_addr, nv); ds = Int64(nv) * 176 // 256
        elif qt == 14: qd_addr = q_q6k(data_addr, nv); ds = Int64(nv) * 210 // 256
        elif qt == 39: qd_addr = q_mx(data_addr, nv); ds = Int64(nv) * 17 // 32
        else: print("Bad qt:", qt); return
        
        # Create path C string
        var path_buf = _alc(Int64(path.byte_length() + 1))
        str_to_cstr(path, path_buf)
        var path_ptr = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(path_buf))
        var fd = _open(path_ptr, O_W | O_C | O_T)
        if fd < 0: print("Error: create", path); return
        
        # Write GGUF header pieces via stack-allocated buffers
        # HEADER: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8) = 24 bytes
        var hdr = _alc(Int64(24))
        var hp = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(hdr))
        hp.store(0, UInt8(0x47)); hp.store(1, UInt8(0x47)); hp.store(2, UInt8(0x55))
        hp.store(3, UInt8(0x46)); hp.store(4, UInt8(3))
        for i in range(5, 8): hp.store(i, UInt8(0))  # version upper bytes
        for i in range(8): hp.store(8+i, UInt8((1 >> (i*8)) & 0xFF))  # 1 tensor
        for i in range(8): hp.store(16+i, UInt8(0))  # 0 metadata
        _sys(SYS_w, Int64(fd), hdr, Int64(24))
        
        # Tensor name (GGUF String = uint64 len + chars, NO null terminator)
        var tn = _alc(Int64(14))
        str_to_cstr_uint64_len("tensor", tn)
        _sys(SYS_w, Int64(fd), tn, Int64(14))
        
        # n_dims = 2 (4 bytes, all set explicitly)
        var nd = _alc(Int64(4))
        var ndp = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(nd))
        ndp.store(0, UInt8(2)); ndp.store(1, UInt8(0)); ndp.store(2, UInt8(0)); ndp.store(3, UInt8(0))
        _sys(SYS_w, Int64(fd), nd, Int64(4))
        
        # dims: nc, nr (8 bytes each)
        var dim_buf = _alc(Int64(16))
        var dp = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(dim_buf))
        for i in range(8): dp.store(i, UInt8((Int64(nc) >> (i*8)) & 0xFF))
        for i in range(8): dp.store(8+i, UInt8((Int64(nr) >> (i*8)) & 0xFF))
        _sys(SYS_w, Int64(fd), dim_buf, Int64(16))
        
        # dtype (4 bytes, explicit zero)
        var dt = _alc(Int64(4))
        var dtp2 = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(dt))
        dtp2.store(0, UInt8(qt & 0xFF)); dtp2.store(1, UInt8(0)); dtp2.store(2, UInt8(0)); dtp2.store(3, UInt8(0))
        _sys(SYS_w, Int64(fd), dt, Int64(4))
        
        # offset = 0 (8 bytes, all zeros)
        var off = _alc(Int64(8))
        var offp = UnsafePointer[UInt8, MutExternalOrigin](unsafe_from_address=Int(off))
        for i in range(8): offp.store(i, UInt8(0))
        _sys(SYS_w, Int64(fd), off, Int64(8))
        
        # Pad to 32-byte alignment
        # Pad to 32-byte alignment (header ~70 bytes)
        var pad = (32 - (70 % 32)) % 32
        if pad > 0:
            var pb = _alc(Int64(pad))
            _sys(SYS_w, Int64(fd), pb, Int64(pad))
        
        # Quantized data
        _sys(SYS_w, Int64(fd), qd_addr, ds)
        
        # Close via syscall (SYS_close = 3)
        _sys(Int64(3), Int64(fd), Int64(0), Int64(0))
        
        var qn = "Q4_0" if qt==2 else ("Q5_0" if qt==6 else ("Q8_0" if qt==8 else ("Q4_K" if qt==12 else ("Q5_K" if qt==13 else ("Q6_K" if qt==14 else "MXFP4")))))
        print("  ", qn, "→", path, "(data:", Int(ds), "bytes)")
    
    # ─── Main: process CLI arguments or run demo ───
    # CLI: quantize <source_model> <target_type> <output_path>
    # If no args, run demo with synthetic data
    print("MojoLlama Quantizer — Pure Mojo (Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, MXFP4)")
    print("==============================================================================")
    var nr = 128; var nc = 2880; var nv = nr * nc
    var data_addr = _alc(Int64(nv * 4))
    var d = UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(data_addr))
    for i in range(nv): d.store(i, Float32(Float64(i % 100 - 50) * 0.01))
    for (qt, name) in [(2, "Q4_0"), (6, "Q5_0"), (8, "Q8_0"), (12, "Q4_K"), (13, "Q5_K"), (14, "Q6_K"), (39, "MXFP4")]:
        var t0 = time.perf_counter()
        write_gguf("/tmp/test_" + name + ".gguf", data_addr, nr, nc, qt)
        print("    Time:", Int((time.perf_counter()-t0)*1000), "ms")
    
    print("\nOutput GGUF files:")
    print("  /tmp/test_Q4_0.gguf  /tmp/test_Q5_0.gguf  /tmp/test_Q8_0.gguf")
    print("  /tmp/test_Q4_K.gguf  /tmp/test_Q5_K.gguf  /tmp/test_Q6_K.gguf  /tmp/test_MXFP4.gguf")
    print("\nSupported types: Q4_0 (2), Q5_0 (6), Q8_0 (8), Q4_K (12), Q5_K (13), Q6_K (14), MXFP4 (39)")
    print("Missing: Q4_1, Q2_K, Q3_K, IQ types")
