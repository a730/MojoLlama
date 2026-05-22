# test_perf.mojo — Test @extern for performance helpers
# Tests if custom .so functions resolve via LD_PRELOAD

@extern("_ml_memalign")
def _ml_alloc(sz: Int, align: Int) abi("C") -> Int: ...

@extern("_ml_free")
def _ml_dealloc(p: Int) abi("C") -> None: ...

@extern("_ml_prefetch_L1")
def _ml_pf_L1(addr: Int) abi("C") -> None: ...

@extern("_ml_pin_process")
def _ml_pin_proc() abi("C") -> None: ...

def main():
    print("Testing perf helper @extern...")
    var p = _ml_alloc(1024, 64)
    print("  _ml_alloc: address =", p)
    _ml_pf_L1(p)
    print("  _ml_prefetch_L1: OK")
    _ml_pin_proc()
    print("  _ml_pin_process: OK")
    _ml_dealloc(p)
    print("  All @extern resolved OK!")
