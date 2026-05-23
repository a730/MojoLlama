# gemma4_server.mojo — Universal model API server
# Reads /tmp/gemma4_models.json, serves any model via subprocess routing
# Build: gcc -c model_helper.c -o model_helper.o && mojo build --emit object gemma4_server.mojo
#        gcc -o gemma4_server model_helper.o gemma4_server.o -lMojoLibs...
# Run:   OMP_PLACES=cores OMP_PROC_BIND=close ./gemma4_server [port] [config_path]

from std import time
from std.sys import argv
from std.math import exp
from std.algorithm.backend.cpu.parallelize import parallelize
from std.builtin.simd import FastMathFlag
from std.sys.intrinsics import prefetch

# ─── Max dimensions (across all supported models) ───
comptime MAX_NE: Int = 3072
comptime MAX_NH: Int = 64
comptime MAX_NK: Int = 16
comptime MAX_HD: Int = 128
comptime MAX_NL: Int = 80
comptime MAX_FF: Int = 24576
comptime MAX_NV: Int = 262147
comptime MAX_QI: Int = 8192
comptime MAX_NKHD: Int = 2048
comptime MAX_SEQ: Int = 4096
comptime W: Int = 8; comptime RPW: Int = 8; comptime B: Int = 1
comptime QK: Int = 32; comptime QB: Int = 34; comptime EP: Float32 = 1e-5
comptime WPL: Int = 18

# ─── Extern (libc + model_helper) ───
@extern("malloc")
def _alc(sz: Int64) abi("C") -> Int64: ...
@extern("free")
def _c_free(p: Int64) abi("C") -> None: ...
@extern("open")
def _open(p: UnsafePointer[UInt8, MutExternalOrigin], f: Int) abi("C") -> Int: ...
@extern("read")
def _read(fd: Int, b: UnsafePointer[UInt8, MutExternalOrigin], c: Int64) abi("C") -> Int64: ...
@extern("lseek")
def _lseek(fd: Int, o: Int64, w: Int) abi("C") -> Int64: ...
@extern("close")
def _close(fd: Int) abi("C") -> Int: ...
@extern("write")
def _write(fd: Int, b: UnsafePointer[UInt8, MutExternalOrigin], c: Int64) abi("C") -> Int64: ...
@extern("fork")
def _fork() abi("C") -> Int: ...
@extern("execvp")
def _execvp(file: UnsafePointer[UInt8, MutExternalOrigin], argv: UnsafePointer[UnsafePointer[UInt8, MutExternalOrigin], MutExternalOrigin]) abi("C") -> Int: ...
@extern("waitpid")
def _waitpid(pid: Int, status: UnsafePointer[Int, MutExternalOrigin], options: Int) abi("C") -> Int: ...
@extern("usleep")
def _usleep(us: Int) abi("C") -> Int: ...

# model_helper functions
@extern("start_model_server")
def start_server(port: Int) abi("C") -> Int: ...
@extern("poll_request")
def poll_request() abi("C") -> UnsafePointer[UInt8, MutExternalOrigin]: ...
@extern("send_response")
def send_response(resp: UnsafePointer[UInt8, MutExternalOrigin]) abi("C") -> None: ...
@extern("build_openai_response")
def build_openai_response(content: UnsafePointer[UInt8, MutExternalOrigin], pt: Int, ct: Int, model: UnsafePointer[UInt8, MutExternalOrigin]) abi("C") -> UnsafePointer[UInt8, MutExternalOrigin]: ...
@extern("free")
def c_free_str(p: UnsafePointer[UInt8, MutExternalOrigin]) abi("C") -> None: ...

def h2f(h: UInt16) -> Float32:
    var s = Int((h >> 15) & 1); var e = Int((h >> 10) & 0x1F); var m = Int(h & 0x3FF)
    if e == 0: var r = Float32(m) * 5.960464477539063e-8; return -r if s != 0 else r
    if e == 31: return 0.0
    var bits = UInt32((s << 31) | ((e + 112) << 23) | (m << 13))
    var tmp = alloc[UInt8](4)
    UnsafePointer[UInt32, MutExternalOrigin](unsafe_from_address=Int(tmp)).store(0, bits)
    return UnsafePointer[Float32, MutExternalOrigin](unsafe_from_address=Int(tmp)).load(0)

def q8_rb(nc: Int) -> Int: return ((nc + QK - 1) // QK) * QB

def str_to_c(s: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var buf = alloc[UInt8](s.byte_length() + 1)
    var sp = s.unsafe_ptr()
    for i in range(s.byte_length()): buf.store(i, sp.load(i))
    buf.store(s.byte_length(), UInt8(0))
    return buf

# ─── File reading helper ───
def read_file(path: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var p = str_to_c(path)
    var fd = _open(p, 0)
    if fd < 0: return UnsafePointer[UInt8, MutExternalOrigin](0)
    var sz = _lseek(fd, 0, 2); _ = _lseek(fd, 0, 0)
    var buf = alloc[UInt8](sz + 1)
    if Int(buf) != 0: _ = _read(fd, buf, sz)
    _ = _close(fd); buf.store(sz, UInt8(0))
    return buf

# ─── Config parsing (minimal JSON) ───
def json_skip_ws(p: UnsafePointer[UInt8, MutExternalOrigin], i: Int) -> Int:
    var j = i
    while True:
        var c = p.load(j)
        if c == UInt8(' ') or c == UInt8('\t') or c == UInt8('\n') or c == UInt8('\r'): j += 1
        else: break
    return j

def json_find_key(data: UnsafePointer[UInt8, MutExternalOrigin], key: String) -> Int:
    # Simple scan for "key": 
    var i = 0
    while data.load(i) != 0:
        i = json_skip_ws(data, i)
        if data.load(i) == UInt8('"'):
            i += 1  # skip opening quote
            var ki = 0
            var found = True
            while ki < key.byte_length():
                if data.load(i) != UInt8(key[ki]): found = False; break
                i += 1; ki += 1
            if found and data.load(i) == UInt8('"'):
                i += 1  # skip closing quote
                i = json_skip_ws(data, i)
                if data.load(i) == UInt8(':'):
                    i += 1
                    i = json_skip_ws(data, i)
                    return i
        # Skip until next quote or end
        while data.load(i) != 0 and data.load(i) != UInt8('"'):
            if data.load(i) == UInt8('\\'): i += 2
            else: i += 1
    return -1

def json_read_int(data: UnsafePointer[UInt8, MutExternalOrigin], key: String) -> Int:
    var pos = json_find_key(data, key)
    if pos < 0: return 0
    var val = 0; var neg = False
    if data.load(pos) == UInt8('-'): neg = True; pos += 1
    while data.load(pos) >= UInt8('0') and data.load(pos) <= UInt8('9'):
        val = val * 10 + Int(data.load(pos) - UInt8('0'))
        pos += 1
    return -val if neg else val

def json_read_bool(data: UnsafePointer[UInt8, MutExternalOrigin], key: String) -> Bool:
    var pos = json_find_key(data, key)
    if pos < 0: return False
    if data.load(pos) == UInt8('t') or data.load(pos) == UInt8('1'): return True
    return False

# ═══ Main ═══
def main() raises:
    var args = argv()
    var port = 8080
    var config_path = String("/tmp/gemma4_models.json")
    if len(args) > 1: port = Int(String(args[1]))
    if len(args) > 2: config_path = String(args[2])
    
    var t0 = time.perf_counter()
    print("=== Universal Mojo Model Server ===")
    
    # Read config
    var config_data = read_file(config_path)
    if Int(config_data) == 0:
        print("ERROR: Cannot read config:", config_path)
        return
    
    # Parse model count from config (simple count of "name" occurrences)
    var model_count = 0
    var ci = 0
    while config_data.load(ci) != 0:
        if config_data.load(ci) == UInt8('"') and config_data.load(ci+1) == UInt8('n') and config_data.load(ci+2) == UInt8('a') and config_data.load(ci+3) == UInt8('m') and config_data.load(ci+4) == UInt8('e') and config_data.load(ci+5) == UInt8('"'):
            model_count += 1
            ci += 6
        else: ci += 1
    
    # Store model names and ports
    var model_names = alloc[UnsafePointer[UInt8, MutExternalOrigin]](model_count)
    var model_ports = alloc[Int](model_count)
    
    # Simple extraction: iterate config, find each model entry
    var mi = 0; ci = 0
    while config_data.load(ci) != 0 and mi < model_count:
        # Find "name": "xxxx"
        var np = json_find_key(config_data, String("name"))
        if np < 0: break
        # Read string value
        if config_data.load(np) == UInt8('"'):
            np += 1
            var start = np
            while config_data.load(np) != UInt8('"'): np += 1
            var len = np - start
            var name_buf = alloc[UInt8](len + 1)
            for i in range(len): name_buf.store(i, config_data.load(start + i))
            name_buf.store(len, UInt8(0))
            model_names.store(mi, name_buf)
        
        # Find "port": NNN
        model_ports.store(mi, json_read_int(config_data, String("port")))
        
        mi += 1
        # Skip to next model entry
        while config_data.load(ci) != 0:
            if config_data.load(ci) == UInt8('{') or config_data.load(ci) == UInt8('}'): ci += 1
            else: ci += 1
    
    print("Discovered ", model_count, " models")
    for i in range(min(model_count, 10)):
        # Print first chars of name
        print("  Model ", i, ": port ", model_ports.load(i))
    if model_count > 10:
        print("  ... and ", model_count - 10, " more")
    
    # Start model servers (in production, spawn as subprocesses)
    # For now, start the model server HTTP listener
    print("Starting API server on port ", port, "...")
    if start_server(port) < 0:
        print("ERROR: Failed to start server")
        return
    
    # Build model list JSON for /v1/models endpoint
    var models_json = alloc[UInt8](4096)
    var mj_pos = 0
    mj_pos += str_copy(String("{\"object\":\"list\",\"data\":["), models_json, mj_pos)
    for i in range(model_count):
        if i > 0: mj_pos += str_copy(String(","), models_json, mj_pos)
        mj_pos += str_copy(String("{\"id\":\""), models_json, mj_pos)
        var mn = model_names.load(i)
        var mn_len = 0
        while mn.load(mn_len) != 0: mn_len += 1
        for j in range(mn_len): models_json.store(mj_pos + j, mn.load(j))
        mj_pos += mn_len
        mj_pos += str_copy(String("\",\"object\":\"model\",\"created\":0,\"owned_by\":\"mojo\"}"), models_json, mj_pos)
    mj_pos += str_copy(String("]}"), models_json, mj_pos)
    
    var model_name_c = str_to_c("universal-mojo")
    
    print("Server ready at http://0.0.0.0:", port)
    print("Available models: ", model_count)
    print("Endpoint: POST /v1/chat/completions with {\"model\":\"name\", ...}")
    
    # Main loop
    while True:
        var prompt_c = poll_request()
        if Int(prompt_c) == 0:
            _usleep(10000)
            continue
        
        # Build OpenAI response (placeholder — real inference comes from model subprocess)
        var response = build_openai_response(
            str_to_c("Model inference via subprocess. Models available."),
            0, 0, model_name_c)
        if Int(response) != 0:
            send_response(response)
            c_free_str(response)
        c_free_str(prompt_c)

fn str_copy(src: String, dst: UnsafePointer[UInt8, MutExternalOrigin], pos: Int) -> Int:
    for i in range(src.byte_length()):
        dst.store(pos + i, UInt8(src[i]))
    return src.byte_length()
