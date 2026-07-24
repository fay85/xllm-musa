#!/usr/bin/env python3
"""Bisect xLLM Mate GDN prefill prep vs mate.gdn_prefill.chunk_gated_delta_rule."""
from __future__ import annotations

import inspect
import sys

sys.path.insert(0, "/workspace/tilelang_musa")
sys.path.insert(0, "/workspace/mate")

import torch
import torch.nn.functional as F
from mate.gdn_prefill import chunk_gated_delta_rule


def xllm_style_cumsum_log_alpha(g_log: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
    alpha = g_log.float().exp().contiguous()
    batch_size, seq_len, num_heads = alpha.shape
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad:
        alpha = F.pad(alpha, (0, 0, 0, pad), value=1.0)
    padded_len = alpha.shape[1]
    log_alpha = alpha.clamp_min(1e-20).log()
    log_alpha = log_alpha.reshape(
        batch_size, padded_len // chunk_size, chunk_size, num_heads
    ).cumsum(2)
    return log_alpha.reshape(batch_size, padded_len, num_heads)


def pad_time_4d(x: torch.Tensor, pad: int) -> torch.Tensor:
    return F.pad(x, (0, 0, 0, 0, 0, pad)) if pad else x


def recurrent_ref(q, k, v, g_log, beta, use_l2=True):
    seq_len, num_q_heads, head_dim = q.shape
    num_v_heads = v.shape[1]
    repeat = num_v_heads // num_q_heads
    query = q.repeat_interleave(repeat, 1).float()
    key = k.repeat_interleave(repeat, 1).float()
    value = v.float()
    g = g_log.float()
    beta_f = beta.float()
    if use_l2:
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)
    query = query * (head_dim**-0.5)
    state = torch.zeros(
        num_v_heads, head_dim, head_dim, device=q.device, dtype=torch.float32
    )
    outs = []
    for t in range(seq_len):
        g_t = g[t].exp().view(num_v_heads, 1, 1)
        b_t = beta_f[t].view(num_v_heads, 1)
        q_t, k_t, v_t = query[t], key[t], value[t]
        state = state * g_t
        kv = (state * k_t.unsqueeze(-1)).sum(-2)
        delta = (v_t - kv) * b_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outs.append((state * q_t.unsqueeze(-1)).sum(-2))
    return torch.stack(outs, 0), state


def main() -> None:
    print("mate.gdn_prefill:", inspect.getfile(chunk_gated_delta_rule))
    print("signature:", inspect.signature(chunk_gated_delta_rule))
    supports_log = "is_log_space" in inspect.signature(chunk_gated_delta_rule).parameters

    torch.manual_seed(0)
    device = "musa"
    seq_len, num_q_heads, num_v_heads, head_dim = 100, 16, 48, 128
    q = torch.randn(seq_len, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(seq_len, num_q_heads, head_dim, device=device, dtype=torch.bfloat16)
    v = torch.randn(seq_len, num_v_heads, head_dim, device=device, dtype=torch.bfloat16)
    g_log = -torch.rand(seq_len, num_v_heads, device=device, dtype=torch.float32) * 0.5
    beta = torch.sigmoid(
        torch.randn(seq_len, num_v_heads, device=device, dtype=torch.float32)
    )
    cu = torch.tensor([0, seq_len], device=device, dtype=torch.int64)
    pad = (64 - seq_len % 64) % 64
    print(f"T={seq_len} pad={pad} supports_is_log_space={supports_log}")

    kwargs = dict(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        beta=beta.contiguous(),
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=True,
    )
    if supports_log:
        o_ref, ht_ref = chunk_gated_delta_rule(
            g=g_log.contiguous(), is_log_space=True, **kwargs
        )
    else:
        o_ref, ht_ref = chunk_gated_delta_rule(g=g_log.exp().contiguous(), **kwargs)

    o_rec, ht_rec = recurrent_ref(q, k, v, g_log, beta)
    print(
        "mate vs recurrent out maxdiff:",
        float((o_ref.float() - o_rec).abs().max()),
        "mean:",
        float((o_ref.float() - o_rec).abs().mean()),
    )
    ht_mate_as_kv = ht_ref[0].transpose(-1, -2)
    print(
        "state layout probe (transpose->[H,K,V]) maxdiff:",
        float((ht_mate_as_kv - ht_rec).abs().max()),
        "no-transpose maxdiff:",
        float((ht_ref[0] - ht_rec).abs().max()),
        "ht shape:",
        tuple(ht_ref.shape),
    )

    cu_pad = torch.tensor([0, seq_len + pad], device=device, dtype=torch.int64)
    q_p = pad_time_4d(q.unsqueeze(0), pad)[0].contiguous()
    k_p = pad_time_4d(k.unsqueeze(0), pad)[0].contiguous()
    v_p = pad_time_4d(v.unsqueeze(0), pad)[0].contiguous()
    g_p = F.pad(g_log, (0, 0, 0, pad), value=0.0).contiguous()
    b_p = F.pad(beta, (0, 0, 0, pad), value=0.0).contiguous()
    kwargs_pad = dict(
        q=q_p,
        k=k_p,
        v=v_p,
        beta=b_p,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=cu_pad,
        use_qk_l2norm_in_kernel=True,
    )
    if supports_log:
        o_pad, ht_pad = chunk_gated_delta_rule(g=g_p, is_log_space=True, **kwargs_pad)
    else:
        o_pad, ht_pad = chunk_gated_delta_rule(g=g_p.exp(), **kwargs_pad)
    print(
        "pad-as-real vs ref out maxdiff:",
        float((o_pad[:seq_len].float() - o_ref.float()).abs().max()),
        "ht maxdiff:",
        float((ht_pad - ht_ref).abs().max()),
    )
    print("xllm host g_cumsum shape:", tuple(xllm_style_cumsum_log_alpha(g_log.unsqueeze(0)).shape))


if __name__ == "__main__":
    main()
