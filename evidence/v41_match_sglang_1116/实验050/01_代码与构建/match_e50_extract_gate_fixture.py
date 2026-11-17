from pathlib import Path
import hashlib, json
from safetensors import safe_open
from safetensors.torch import save_file

root = Path('/work/artifacts/model_view_w4afp8_mt_1017')
out = Path('/expt/inputs/e50_real_gate_weights.safetensors')
assert not out.exists()
index = json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
tensors = {}
sources = {}
for key in ['layers.0.ffn.gate.weight', 'layers.0.ffn.gate.bias', 'mtp.0.ffn.gate.weight', 'mtp.0.ffn.gate.bias']:
    path = root/index[key]
    with safe_open(path, framework='pt', device='cpu') as f:
        tensor = f.get_tensor(key)
    tensors[key] = tensor
    sources[key] = {'file': str(path), 'shape': list(tensor.shape), 'dtype': str(tensor.dtype), 'tensor_sha256': hashlib.sha256(tensor.contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest()}
save_file(tensors, str(out))
config = json.loads((root/'config.json').read_text())
meta = {'fixture': str(out), 'sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'bytes': out.stat().st_size, 'sources': sources, 'config': config}
out.with_suffix('.json').write_text(json.dumps(meta, indent=2))
print(json.dumps({k: meta[k] for k in ['fixture','sha256','bytes','sources']}))
