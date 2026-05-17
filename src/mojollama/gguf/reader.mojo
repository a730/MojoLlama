from std.python import Python, PythonObject
from .types import *

struct GGUFFile:
    var data: PythonObject
    var offset: PythonObject
    var alignment: UInt64

    fn __init__(out self, path: String) raises:
        self.data = Python.evaluate("open")(path, "rb").read()
        self.offset = Python.evaluate("0")
        self.alignment = 32

    fn read_slice(mut self, n: PythonObject) raises -> PythonObject:
        var py_slice = Python.evaluate("slice")(self.offset, self.offset + n)
        var result = self.data[py_slice]
        self.offset = self.offset + n
        return result

    fn read_py_uint32(mut self) raises -> PythonObject:
        var b = self.read_slice(Python.evaluate("4"))
        return Python.evaluate("int.from_bytes")(b, "little")

    fn read_py_uint64(mut self) raises -> PythonObject:
        var b = self.read_slice(Python.evaluate("8"))
        return Python.evaluate("int.from_bytes")(b, "little")

    fn read_uint32(mut self) raises -> UInt32:
        return UInt32(py=self.read_py_uint32())

    fn read_uint64(mut self) raises -> UInt64:
        return UInt64(py=self.read_py_uint64())

    fn read_string(mut self) raises -> PythonObject:
        var length = self.read_py_uint64()
        var raw = self.read_slice(length)
        return raw.decode("utf-8")

fn load_gguf(path: String) raises -> PythonObject:
    var f = GGUFFile(path)

    var magic = f.read_uint32()
    if magic != GGUF_MAGIC:
        print("Invalid GGUF magic:", magic)
        return Python.evaluate("None")

    var version = f.read_uint32()
    if version != GGUF_VERSION:
        print("Unsupported GGUF version:", version)
        return Python.evaluate("None")

    var tensor_count = Int(py=f.read_py_uint64())
    var metadata_kv_count = Int(py=f.read_py_uint64())

    var metadata = Python.evaluate("{}")
    var struct_mod = Python.evaluate("__import__('struct')")

    for i in range(metadata_kv_count):
        var key = f.read_string()
        var value_type = f.read_uint32()
        var value: PythonObject

        if value_type == GGUFValueType.STRING:
            value = f.read_string()
        elif value_type == GGUFValueType.UINT32 or value_type == GGUFValueType.INT32:
            value = f.read_py_uint32()
        elif value_type == GGUFValueType.UINT64 or value_type == GGUFValueType.INT64:
            value = f.read_py_uint64()
        elif value_type == GGUFValueType.FLOAT32:
            var bits = f.read_py_uint32()
            var b4 = struct_mod.pack("I", bits)
            value = struct_mod.unpack("f", b4)[0]
        elif value_type == GGUFValueType.FLOAT64:
            var bits_lo = f.read_py_uint32()
            var bits_hi = f.read_py_uint32()
            var b8 = struct_mod.pack("II", bits_lo, bits_hi)
            value = struct_mod.unpack("d", b8)[0]
        elif value_type == GGUFValueType.BOOL:
            var byte_slice = f.read_slice(Python.evaluate("1"))
            value = Python.evaluate("bool")(byte_slice[0])
        elif value_type == GGUFValueType.ARRAY:
            var arr_type = f.read_uint32()
            var arr_len = Int(py=f.read_py_uint64())
            var arr = Python.evaluate("[]")
            for j in range(arr_len):
                if arr_type == GGUFValueType.STRING:
                    arr.append(f.read_string())
                elif arr_type == GGUFValueType.UINT32 or arr_type == GGUFValueType.INT32:
                    arr.append(f.read_py_uint32())
                elif arr_type == GGUFValueType.FLOAT32:
                    var bits2 = f.read_py_uint32()
                    var b4_2 = struct_mod.pack("I", bits2)
                    arr.append(struct_mod.unpack("f", b4_2)[0])
                elif arr_type == GGUFValueType.UINT64 or arr_type == GGUFValueType.INT64:
                    arr.append(f.read_py_uint64())
            value = arr
        else:
            value = Python.evaluate("None")

        metadata[key] = value

    var alignment = GGUF_DEFAULT_ALIGNMENT
    if metadata.__contains__("general.alignment"):
        alignment = UInt64(py=metadata["general.alignment"])
    f.alignment = alignment

    var tensor_info_list = Python.evaluate("[]")
    for i in range(tensor_count):
        var name = f.read_string()
        var n_dim = Int(py=f.read_py_uint32())
        var dims = Python.evaluate("[]")
        for j in range(n_dim):
            dims.append(f.read_py_uint64())
        var dtype = f.read_uint32()
        var offset = f.read_py_uint64()

        var n_elems: UInt64 = 1
        for j in range(n_dim):
            n_elems *= UInt64(py=dims[j])
        var size_bytes = ggml_type_size_in_bytes(dtype, n_elems)

        var t = Python.evaluate("{}")
        t["name"] = name
        t["n_dimensions"] = n_dim
        t["dimensions"] = dims
        t["dtype"] = dtype
        t["offset"] = offset
        t["size_bytes"] = size_bytes
        tensor_info_list.append(t)

    metadata["_tensor_count"] = tensor_count
    metadata["_tensor_info"] = tensor_info_list
    metadata["_alignment"] = alignment
    metadata["_tensor_data_offset"] = f.offset

    return metadata
