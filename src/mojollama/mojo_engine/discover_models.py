#!/usr/bin/env python3
"""Discover models and generate engine config for the Mojo API server."""
import json, os, glob, subprocess

def scan_models():
    """Scan all weight directories and build model registry."""
    models = []
    
    # Scan /tmp/weights_* for arch.json
    for d in sorted(glob.glob('/tmp/weights_*')):
        arch_path = os.path.join(d, 'arch.json')
        if not os.path.exists(arch_path):
            continue
        with open(arch_path) as f:
            arch = json.load(f)
        
        name = os.path.basename(d).replace('weights_', '')
        
        # Determine which engine binary to use
        engine = None
        if arch.get('has_qk_norm') and arch.get('has_inp_gate'):
            engine = 'gemma4_gen_q8'  # Gemma 4 models
        elif arch.get('hd') == 64:
            engine = 'tinyllama_gen_q8'  # TinyLlama family
        elif arch.get('has_qk_norm'):
            engine = 'gemma4_gen_q8'  # Qwen3, etc (use Gemma engine subset)
        else:
            engine = 'gemma4_gen_q8'  # Default fallback
        
        models.append({
            'name': name,
            'arch': arch,
            'weights': d,
            'engine': engine,
            'port': 9000 + len(models)  # Assign internal port
        })
    
    return models

if __name__ == '__main__':
    models = scan_models()
    config = {'models': []}
    
    print(f"=== Discovered {len(models)} Models ===\n")
    for m in models:
        a = m['arch']
        status = '✓ ready' if a['ne'] > 0 else '✗ incomplete'
        print(f"  {m['name']}:")
        print(f"    NE={a['ne']} NH={a['nh']} NK={a['nk']} HD={a['hd']}")
        print(f"    NL={a['nl']} FF={a['ff']} NV={a['nv']}")
        print(f"    QK-norm={a['has_qk_norm']} inp_gate={a['has_inp_gate']} RoPE={a['has_rope']}")
        print(f"    Engine: {m['engine']} Port: {m['port']} {status}")
        print()
        
        config['models'].append({
            'name': m['name'],
            'display_name': m['name'].replace('-', ' ').title(),
            'engine': m['engine'],
            'weights': m['weights'],
            'port': m['port'],
            'arch': a
        })
    
    # Write config
    with open('/tmp/gemma4_models.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config written to /tmp/gemma4_models.json ({len(models)} models)")
