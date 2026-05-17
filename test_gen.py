"""Test generation pipeline."""
import sys, time, gc
sys.path.insert(0, 'src')
import numpy as np
from mojollama.model.inference import load_model

model = load_model('test_model.gguf')
ids = model.encode('Hello')

t0 = time.time()
logits = model.forward(ids)
print(f'Forward 1: {time.time()-t0:.1f}s')
top = int(np.argmax(logits[-1]))
print(f'Top token: {top}={model.decode([top])!r}')

ids.append(top)
gc.collect()
t0 = time.time()
logits2 = model.forward(ids)
print(f'Forward 2: {time.time()-t0:.1f}s')
top2 = int(np.argmax(logits2[-1]))
print(f'Top token: {top2}={model.decode([top2])!r}')

ids.append(top2)
gc.collect()
t0 = time.time()
logits3 = model.forward(ids)
print(f'Forward 3: {time.time()-t0:.1f}s')

gc.collect()
print('Full generation:')
t0 = time.time()
result = model.generate('The capital of France is', max_tokens=5)
print(f'Generate: {time.time()-t0:.1f}s')
print('Result:', repr(result))
