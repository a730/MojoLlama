from std.python import Python
from mojollama.gguf.reader import load_gguf

fn main() raises:
    var model_path_py = Python.evaluate("__import__('os').environ.get('MODEL_PATH', '')")

    if Python.evaluate("len")(model_path_py) == 0:
        var glob = Python.evaluate("__import__('glob')")
        var files = glob.glob("*.gguf")
        if Python.evaluate("len")(files) > 0:
            model_path_py = files[0]
        else:
            print("Usage: MODEL_PATH=<model.gguf> ./build/mojollama")
            return

    var model_path = String(model_path_py)
    print("Loading GGUF:", model_path)
    var metadata = load_gguf(model_path)

    var is_none = Python.evaluate("lambda x: x is None")(metadata)
    if is_none:
        print("Failed to load model")
        return

    var arch = "unknown"
    if metadata.__contains__("general.architecture"):
        arch = String(metadata["general.architecture"])
    print("Architecture:", arch)

    var tensor_info = metadata["_tensor_info"]
    var n_tensors = Int(py=Python.evaluate("len")(tensor_info))
    print("Tensors:", n_tensors)

    if n_tensors > 0:
        var t0 = tensor_info[0]
        print("First tensor:", String(t0["name"]))

    print("Done")
