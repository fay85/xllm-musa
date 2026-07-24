#!/usr/bin/env python3
import os, sys, ctypes
sys.path.insert(0, "/workspace/tilelang_musa")
import torch, torch.nn.functional as F
import tvm_ffi

for lib in ("/usr/local/musa/lib/libmusart.so",):
    if os.path.exists(lib):
        ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)

def host_cumsum(g_log, cs=64):
    # new xLLM path: direct log cumsum, slice back
    x = g_log.float().contiguous()
    B,T,H = x.shape
    pad = (cs - T % cs) % cs
    if pad:
        x = F.pad(x, (0,0,0,pad), value=0.0)
    Tp = x.shape[1]
    y = x.reshape(B, Tp//cs, cs, H).cumsum(2).reshape(B, Tp, H)
    return y[:, :T].contiguous()

def recurrent(q,k,v,g,beta):
    B,T,Hq,D=q.shape; Hv=v.shape[2]; r=Hv//Hq
    q=q.repeat_interleave(r,2).float(); k=k.repeat_interleave(r,2).float()
    v=v.float(); g=g.float(); beta=beta.float()
    q=F.normalize(q,2,-1); k=F.normalize(k,2,-1); q=q*(D**-0.5)
    h=torch.zeros(B,Hv,D,D,device=q.device,dtype=torch.float32)
    outs=[]
    for t in range(T):
        gt=g[:,t].exp().view(B,Hv,1,1); bt=beta[:,t].view(B,Hv,1)
        qt,kt,vt=q[:,t],k[:,t],v[:,t]
        h=h*gt; kv=(h*kt.unsqueeze(-1)).sum(-2); delta=(vt-kv)*bt
        h=h+kt.unsqueeze(-1)*delta.unsqueeze(-2)
        outs.append((h*qt.unsqueeze(-1)).sum(-2))
    return torch.stack(outs,1), h

def run_ffi(q,k,v,g,beta):
    uri="mate_gdn_prefill_hq16_hv48_bf16"
    mod=tvm_ffi.load_module(f"/workspace/mate_cached_ops/{uri}/{uri}.so")
    run=mod["run"]
    B,T,Hq,D=q.shape; Hv=v.shape[2]
    qn=F.normalize(q.float(),2,-1).to(q.dtype).contiguous()
    kn=F.normalize(k.float(),2,-1).to(k.dtype).contiguous()
    g_cs=host_cumsum(g)
    a=torch.zeros(B,T,Hv,64,device=q.device,dtype=q.dtype)
    h0=torch.zeros(B,Hv,D,D,device=q.device,dtype=torch.float32)
    o=torch.empty(B,T,Hv,D,device=q.device,dtype=v.dtype)
    ht=torch.empty(B,Hv,D,D,device=q.device,dtype=torch.float32)
    run(qn,kn,v.contiguous(),a,g_cs,beta.float().contiguous(),h0,o,ht)
    return o, ht

def case(name, T, g_scale):
    torch.manual_seed(0)
    device="musa"
    B,Hq,Hv,D=1,16,48,128
    q=torch.randn(B,T,Hq,D,device=device,dtype=torch.bfloat16)
    k=torch.randn(B,T,Hq,D,device=device,dtype=torch.bfloat16)
    v=torch.randn(B,T,Hv,D,device=device,dtype=torch.bfloat16)
    g=-torch.rand(B,T,Hv,device=device,dtype=torch.float32)*g_scale
    beta=torch.sigmoid(torch.randn(B,T,Hv,device=device,dtype=torch.float32))
    o_m,ht_m=run_ffi(q,k,v,g,beta)
    o_r,ht_r=recurrent(q,k,v,g,beta)
    print(f"{name}: T={T} g_scale={g_scale} out_maxdiff={float((o_m.float()-o_r).abs().max()):.6g} "
          f"state_viaT={float((ht_m.transpose(-1,-2)-ht_r).abs().max()):.6g} "
          f"g_min={float(g.min()):.4g} mate_mean={float(o_m.float().abs().mean()):.6g} ref_mean={float(o_r.abs().mean()):.6g}")

if __name__=="__main__":
    case("short_mild", 11, 0.5)
    case("short_harsh", 11, 8.0)
    case("long_harsh", 128, 8.0)
    case("short_exact_e2e_g", 11, 8.0)
