#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

sys.path.insert(0, "/workspace/tilelang_musa")

import torch
import torch.nn.functional as F
import tvm_ffi


def recurrent(q, k, v, g, beta):
    batch_size, seq_len, num_q_heads, head_dim = q.shape
    num_v_heads = v.shape[2]
    repeat = num_v_heads // num_q_heads
    query = q.repeat_interleave(repeat, 2).float()
    key = k.repeat_interleave(repeat, 2).float()
    value = v.float()
    g_f = g.float()
    beta_f = beta.float()
    query = F.normalize(query, 2, -1)
    key = F.normalize(key, 2, -1)
    query = query * (head_dim**-0.5)
    # recurrent state [B, H, K, V]
    state = torch.zeros(
        batch_size,
        num_v_heads,
        head_dim,
        head_dim,
        device=q.device,
        dtype=torch.float32,
    )
    outs = []
    for t in range(seq_len):
        g_t = g_f[:, t].exp().view(batch_size, num_v_heads, 1, 1)
        b_t = beta_f[:, t].view(batch_size, num_v_heads, 1)
        q_t, k_t, v_t = query[:, t], key[:, t], value[:, t]
        state = state * g_t
        kv = (state * k_t.unsqueeze(-1)).sum(-2)
        delta = (v_t - kv) * b_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((state * q_t.unsqueeze(-1)).sum(-2))
    return torch.stack(outs, 1), state


def host_cumsum(g_log, chunk_size=64):
    alpha = g_log.float().exp()
    batch_size, seq_len, num_heads = alpha.shape
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad:
        alpha = F.pad(alpha, (0, 0, 0, pad), value=1.0)
    padded_len = alpha.shape[1]
    log_alpha = (
        alpha.clamp_min(1e-20)
        .log()
        .reshape(batch_size, padded_len // chunk_size, chunk_size, num_heads)
        .cumsum(2)
        .reshape(batch_size, padded_len, num_heads)
    )
    return log_alpha, pad


def run_case(seq_len: int) -> None:
    torch.manual_seed(0)
    device = "musa"
    batch_size, num_q_heads, num_v_heads, head_dim = 1, 16, 48, 128
    q = torch.randn(
        batch_size, seq_len, num_q_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    k = torch.randn(
        batch_size, seq_len, num_q_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    v = torch.randn(
        batch_size, seq_len, num_v_heads, head_dim, device=device, dtype=torch.bfloat16
    )
    g = -torch.rand(batch_size, seq_len, num_v_heads, device=device, dtype=torch.float32) * 0.5
    beta = torch.sigmoid(
        torch.randn(batch_size, seq_len, num_v_heads, device=device, dtype=torch.float32)
    )

    uri = "mate_gdn_prefill_hq16_hv48_bf16"
    ops = os.environ.get("FLASHINFER_OPS_PATH", "/workspace/mate_cached_ops")
    mod = tvm_ffi.load_module(f"{ops}/{uri}/{uri}.so")
    run = mod["run"]

    qn = F.normalize(q.float(), 2, -1).to(q.dtype).contiguous()
    kn = F.normalize(k.float(), 2, -1).to(k.dtype).contiguous()
    g_cs, pad = host_cumsum(g)
    print(f"T={seq_len} pad={pad}")

    # Variant A: current xLLM (pad qkv to match g_cs length)
    if pad:
        qn_p = F.pad(qn, (0, 0, 0, 0, 0, pad))
        kn_p = F.pad(kn, (0, 0, 0, 0, 0, pad))
        v_p = F.pad(v, (0, 0, 0, 0, 0, pad))
        beta_p = F.pad(beta.float(), (0, 0, 0, pad), value=0.0)
        tokens = seq_len + pad
    else:
        qn_p, kn_p, v_p, beta_p, tokens = qn, kn, v, beta.float(), seq_len

    a = torch.empty(batch_size, tokens, num_v_heads, 64, device=device, dtype=q.dtype)
    h0 = torch.zeros(
        batch_size, num_v_heads, head_dim, head_dim, device=device, dtype=torch.float32
    )
    o = torch.empty(batch_size, tokens, num_v_heads, head_dim, device=device, dtype=v.dtype)
    ht = torch.empty(
        batch_size, num_v_heads, head_dim, head_dim, device=device, dtype=torch.float32
    )
    run(
        qn_p.contiguous(),
        kn_p.contiguous(),
        v_p.contiguous(),
        a,
        g_cs.contiguous(),
        beta_p.contiguous(),
        h0,
        o,
        ht,
    )
    o_ref, h_ref = recurrent(q, k, v, g, beta)
    o_s = o[:, :seq_len]
    print(
        "A pad-qkv out_maxdiff",
        float((o_s.float() - o_ref).abs().max()),
        "mean",
        float((o_s.float() - o_ref).abs().mean()),
        "nan",
        bool(torch.isnan(o_s.float()).any()),
    )
    print(
        "A ht vs [H,K,V] viaT",
        float((ht.transpose(-1, -2) - h_ref).abs().max()),
        "noT",
        float((ht - h_ref).abs().max()),
    )

    # Variant B: no pad on qkv; slice g_cs to T (kernel handles partial chunk)
    g_cs_t = g_cs[:, :seq_len].contiguous()
    a2 = torch.empty(batch_size, seq_len, num_v_heads, 64, device=device, dtype=q.dtype)
    h02 = torch.zeros_like(h0)
    o2 = torch.empty_like(o_ref, dtype=v.dtype)
    ht2 = torch.empty_like(ht)
    run(
        qn.contiguous(),
        kn.contiguous(),
        v.contiguous(),
        a2,
        g_cs_t,
        beta.float().contiguous(),
        h02,
        o2,
        ht2,
    )
    print(
        "B no-pad out_maxdiff",
        float((o2.float() - o_ref).abs().max()),
        "mean",
        float((o2.float() - o_ref).abs().mean()),
        "nan",
        bool(torch.isnan(o2.float()).any()),
    )
    print(
        "B ht viaT",
        float((ht2.transpose(-1, -2) - h_ref).abs().max()),
        "noT",
        float((ht2 - h_ref).abs().max()),
    )

    # Variant C: pass raw log g without host cumsum (wrong if kernel expects cumsum)
    a3 = torch.empty_like(a2)
    h03 = torch.zeros_like(h0)
    o3 = torch.empty_like(o2)
    ht3 = torch.empty_like(ht)
    run(
        qn.contiguous(),
        kn.contiguous(),
        v.contiguous(),
        a3,
        g.float().contiguous(),
        beta.float().contiguous(),
        h03,
        o3,
        ht3,
    )
    print(
        "C raw-log-g out_maxdiff",
        float((o3.float() - o_ref).abs().max()),
        "mean",
        float((o3.float() - o_ref).abs().mean()),
    )

    # Variant D: pass alpha=exp(g) without cumsum
    a4 = torch.empty_like(a2)
    h04 = torch.zeros_like(h0)
    o4 = torch.empty_like(o2)
    ht4 = torch.empty_like(ht)
    run(
        qn.contiguous(),
        kn.contiguous(),
        v.contiguous(),
        a4,
        g.exp().contiguous(),
        beta.float().contiguous(),
        h04,
        o4,
        ht4,
    )
    print(
        "D alpha-no-cumsum out_maxdiff",
        float((o4.float() - o_ref).abs().max()),
        "mean",
        float((o4.float() - o_ref).abs().mean()),
    )


if __name__ == "__main__":
    # Preload MUSA runtime for kernel_lib.so
    import ctypes

    for lib in (
        "/usr/local/musa/lib/libmusart.so",
        "/usr/local/musa/lib/libmusa.so",
    ):
        if os.path.exists(lib):
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            print("preloaded", lib)
    for t in (128, 100, 64):
        print("=" * 60)
        run_case(t)
