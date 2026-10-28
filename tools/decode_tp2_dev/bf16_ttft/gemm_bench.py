import torch

try:
    import torch_musa  # noqa: F401
except Exception as exc:  # pragma: no cover
    print("torch_musa import:", exc)


dev = torch.device("musa:0")
torch.musa.set_device(0)

# (name, N, K) per rank for TP2 prefill; TP1 = 2x N or K
shapes_tp2 = [
    ("gdn_in_proj_qkvz", 8192, 5120),
    ("attn_qkv_gate", 7168, 5120),
    ("mlp_gate_up", 17408, 5120),
    ("mlp_down", 5120, 8704),
    ("gdn_out/o_proj", 5120, 3072),
]
shapes_tp1 = [
    ("mlp_gate_up_tp1", 34816, 5120),
    ("mlp_down_tp1", 5120, 17408),
]

def bench(m, n, k, iters=30):
    a = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    w = torch.randn(n, k, device=dev, dtype=torch.bfloat16)
    out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    for _ in range(5):
        torch.mm(a, w.t(), out=out)
    torch.musa.synchronize()
    start = torch.musa.Event(enable_timing=True)
    end = torch.musa.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        torch.mm(a, w.t(), out=out)
    end.record()
    torch.musa.synchronize()
    ms = start.elapsed_time(end) / iters
    tflops = 2.0 * m * n * k / (ms * 1e-3) / 1e12
    return ms, tflops

print("device", torch.musa.get_device_name(0))
for m in (1024, 4096):
    for name, n, k in shapes_tp2 + shapes_tp1:
        ms, tf = bench(m, n, k)
        print(f"M={m:5d} {name:18s} N={n:6d} K={k:6d}  {ms:7.3f} ms  {tf:7.1f} TFLOPS")

# peak-ish square GEMM
for s in (4096, 8192):
    ms, tf = bench(s, s, s, iters=20)
    print(f"square {s}: {ms:.3f} ms {tf:.1f} TFLOPS")

# per-layer GEMM time for one 1024-token chunk on one TP2 rank
m = 1024
gdn = sum(bench(m, n, k)[0] for name, n, k in shapes_tp2 if name in ("gdn_in_proj_qkvz", "gdn_out/o_proj"))
attn = sum(bench(m, n, k)[0] for name, n, k in shapes_tp2 if name in ("attn_qkv_gate", "gdn_out/o_proj"))
mlp = sum(bench(m, n, k)[0] for name, n, k in shapes_tp2 if name in ("mlp_gate_up", "mlp_down"))
chunk_ms = 48 * (gdn + mlp) + 16 * (attn + mlp)
print(f"TP2 rank, 1024-token chunk: GEMM-only {chunk_ms:.1f} ms (48 GDN layers + 16 attn layers)")
