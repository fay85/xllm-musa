#!/usr/bin/env python3
import ctypes, os, sys
sys.path.insert(0, "/workspace/tilelang_musa")
import torch
import torch.nn.functional as F
import tvm_ffi

ctypes.CDLL("/usr/local/musa/lib/libmusart.so", mode=ctypes.RTLD_GLOBAL)
objs = torch.load("/tmp/mate_gdn_bisect_tensors.pt", map_location="cpu", weights_only=False)
if isinstance(objs, dict):
    objs = list(objs.values())
names = ["q","k","v","g","beta","mate_out","ref_out","mate_state","ref_state"]
for n, t in zip(names, objs):
    print(n, tuple(t.shape), t.dtype)
q, k, v, g, beta, mate_out, ref_out, mate_state, ref_state = objs
print("dump mate vs ref out", float((mate_out.float() - ref_out.float()).abs().max()))

device = "musa"
q, k, v, g, beta = [t.to(device) for t in (q, k, v, g, beta)]

def host_cumsum(g_log, cs=64):
    x = g_log.float().contiguous()
    B, T, H = x.shape
    pad = (cs - T % cs) % cs
    if pad:
        x = F.pad(x, (0, 0, 0, pad), value=0.0)
    Tp = x.shape[1]
    y = x.reshape(B, Tp // cs, cs, H).cumsum(2).reshape(B, Tp, H)
    return y[:, :T].contiguous()

uri = "mate_gdn_prefill_hq16_hv48_bf16"
run = tvm_ffi.load_module(f"/workspace/mate_cached_ops/{uri}/{uri}.so")["run"]
B, T, Hq, D = q.shape
Hv = v.shape[2]
qn = F.normalize(q.float(), 2, -1).to(q.dtype).contiguous()
kn = F.normalize(k.float(), 2, -1).to(k.dtype).contiguous()
g_cs = host_cumsum(g)
a = torch.zeros(B, T, Hv, 64, device=device, dtype=q.dtype)
h0 = torch.zeros(B, Hv, D, D, device=device, dtype=torch.float32)
o = torch.empty(B, T, Hv, D, device=device, dtype=v.dtype)
ht = torch.empty(B, Hv, D, D, device=device, dtype=torch.float32)
run(qn, kn, v.contiguous(), a, g_cs, beta.float().contiguous(), h0, o, ht)
print("replay vs dump mate_out", float((o.cpu().float() - mate_out.float()).abs().max()))
print("replay vs dump ref_out", float((o.cpu().float() - ref_out.float()).abs().max()))

def recurrent(q, k, v, g, beta):
    B, T, Hq, D = q.shape
    Hv = v.shape[2]
    r = Hv // Hq
    q = q.repeat_interleave(r, 2).float()
    k = k.repeat_interleave(r, 2).float()
    v = v.float()
    g = g.float()
    beta = beta.float()
    q = F.normalize(q, 2, -1)
    k = F.normalize(k, 2, -1)
    q = q * (D ** -0.5)
    h = torch.zeros(B, Hv, D, D, device=q.device, dtype=torch.float32)
    outs = []
    for t in range(T):
        gt = g[:, t].exp().view(B, Hv, 1, 1)
        bt = beta[:, t].view(B, Hv, 1)
        qt, kt, vt = q[:, t], k[:, t], v[:, t]
        h = h * gt
        kv = (h * kt.unsqueeze(-1)).sum(-2)
        delta = (vt - kv) * bt
        h = h + kt.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((h * qt.unsqueeze(-1)).sum(-2))
    return torch.stack(outs, 1), h

o_r, h_r = recurrent(q, k, v, g, beta)
print("replay vs recurrent", float((o.float() - o_r).abs().max()))
print("ref vs recurrent", float((ref_out.to(device).float() - o_r).abs().max()))
print("ht vs recurrent viaT", float((ht.transpose(-1, -2) - h_r).abs().max()))
print("ref_state vs recurrent", float((ref_state.to(device).float() - h_r).abs().max()))
