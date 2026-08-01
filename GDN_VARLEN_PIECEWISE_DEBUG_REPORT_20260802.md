# Unpadded Varlen GDN 与 Piecewise Graph 性能定位报告（2026-08-02）

## 结论

本阶段定位并修复了两个独立瓶颈：

1. unpadded varlen GDN 的额外开销在通用 varlen KKT，不在 recurrent core。
2. piecewise graph 的主要剩余损失来自 16 个 dense FA3 attention 边界；原 replay 被拆成 17 graph + 16 runner + 33 instructions。

最终实现：

- C1 unpadded KKT 复用 fixed partial-tail kernel，recurrent 仍是真正 varlen。
- C1 dense FA3 在 padding≤32 的安全域直接进入 surrounding graph，变为 1 graph + 0 runner。
- 超过安全域或 B>1 自动保留 split runner。
- 两项优化均有 rollback，fixed KKT SO 缺失时自动回退 varlen SO。

最终 mixed fixed-20 reverse A/B：

| 模式 | TTFT (ms) | model forward (ms) |
|---|---:|---:|
| adaptive full-piecewise | 266.336 | 233.248 |
| pure eager | 267.851 | 234.447 |
| adaptive - eager | **-1.515** | **-1.199** |

两轮配对均为收益，generated texts exact，FATAL / NaN / ERROR 为 0。

## 1. Varlen GDN 根因

同 shape eager reverse A/B：

`/workspace/bench_results/gdn_varlen_deepdebug_20260801/eager_unpadded_vs_padded_ab`

| 模式 | TTFT (ms) | forward (ms) |
|---|---:|---:|
| unpadded varlen | 268.482 | 235.404 |
| padded | 270.667 | 237.572 |
| unpadded - padded | **-2.185** | **-2.167** |

Matched Kineto：

`/workspace/bench_results/gdn_varlen_deepdebug_20260801/kineto_matched`

| bucket | unpadded (ms) | padded (ms) |
|---|---:|---:|
| recurrent GDN | 7.252 | 7.310 |
| L2 norm | 4.401 | 4.564 |
| KKT | 4.299 | 2.618 |
| CopyLastContiguous | 4.414 | 7.366 |

通用 varlen KKT 对每个 chunk/head 执行 cu_seqlens binary search 和 multi-sequence while 状态机；C1 只有一个 sequence，这些控制流冗余。

MATE fixed KKT 本身已有 `ceildiv(T,64)` 和 `valid_seqs` tail mask。T=2057 standalone、48 launches：

- fixed partial-tail：2.356 ms；
- generic varlen：3.618 ms；
- 改善：1.262 ms。

T=2049/2057/2079 均 bitwise exact、max_abs=0、tail max_abs=0、全 finite。

End-to-end reverse A/B：

`/workspace/bench_results/gdn_varlen_deepdebug_20260801/c1_partial_kkt_ab`

| 模式 | TTFT (ms) | forward (ms) |
|---|---:|---:|
| fixed partial KKT | 266.615 | 233.452 |
| generic varlen | 269.502 | 236.071 |
| fixed - varlen | **-2.887** | **-2.619** |

Matched Kineto `kineto_c1_partial_kkt`：KKT 4.287→2.625 ms；recurrent 7.172→7.140 ms。

## 2. Piecewise 收益为何少

零 padding exact2048：

`/workspace/bench_results/gdn_varlen_deepdebug_20260801/exact2048_piecewise_ab`

| 模式 | TTFT (ms) | forward (ms) |
|---|---:|---:|
| eager | 253.781 | 220.456 |
| split piecewise | 252.902 | 218.864 |
| piecewise - eager | **-0.879** | **-1.592** |

原 capture 为 17 graphs、16 FA3 runners、33 instructions。segment profile：

- graph segments：213.5–215.1 ms；
- attention runners：约 8.2 ms；
- metadata update：约 0.28 ms。

证据：`exact2048_piecewise_profile`。

Matched Kineto `kineto_exact2048_graph_ab`：

- eager：约 2145 次 kernel launch，CPU launch 13.27 ms；
- split piecewise：约 1071 次，CPU launch 6.54 ms；
- 17 次新增 prefill graph launch 约 1.05 ms。

CPU launch 确实下降，但大部分原 launch 已被 device compute 覆盖，所以 wall 只兑现约 1.6 ms。

SGLang 参考实现同时将 `RadixAttention` 和 `RadixLinearAttention` 注册为 split op；xLLM 原实现只拆 16 个 full attention，48 个 GDN 已在 graph 内。问题不是 fake replay，而是 full-attention 边界仍限制收益。

## 3. Full-FA3 capture

安全域内 dense FA3 直接在 surrounding graph capture：

`17 graph + 16 runner + 33 instructions` → `1 graph + 0 runner + 1 instruction`。

Full vs split reverse A/B：

`/workspace/bench_results/gdn_varlen_deepdebug_20260801/full_fa3_capture_ab`

| 模式 | TTFT (ms) | forward (ms) |
|---|---:|---:|
| full FA3 | 250.459 | 216.856 |
| split runner | 252.828 | 218.766 |
| full - split | **-2.369** | **-1.910** |

同一最终实现 binary 的 eager：253.067 / 219.650 ms。因此 full-piecewise 比 eager 快 **2.608 ms TTFT / 2.794 ms forward**。

### Correctness 与负门禁

通过：

- exact2048：40/40 full vs split 文本 exact；
- capture2048→replay2046：5/5 temp0 exact；
- capture2046→replay2048：5/5 temp0 exact；
- production boundary 2080→2080/2081/2096/2112（padding≤32）：5/5 temp0 exact。

证据目录：

- `full_fa3_dynamic2046_correctness`
- `full_fa3_dynamic2048_reverse_correctness`
- `full_fa3_padding32_boundary`

负门禁：`full_fa3_forced2112_correctness`。强制 padding41–62 时 full 与 split temp0 文本不 exact，因此没有放宽阈值。

最终控制流 smoke `safe_full_fa3_final_smokes`：

- default exact：1/0/1；
- rollback exact：17/16/33；
- wide padding fallback：17/16/33；
- 全部完成，错误为 0。

## 4. 最终生产 mixed A/B

证据：

`/workspace/bench_results/gdn_varlen_deepdebug_20260801/final_adaptive_vs_eager_ab`

- adaptive 每轮：17 eager + 3 piecewise；
- pure eager：20 eager；
- 配对 TTFT：-2.583 ms、-0.448 ms；
- 两轮文本 exact，错误为 0。

## 5. 源码、开关与 build

修改：

- `xllm/core/kernels/musa/gdn_prefill.cpp`
- `xllm/core/kernels/musa/attention_runner.cpp`
- `xllm/core/runtime/musa/musa_graph_executor_impl.cpp`

开关：

- `XLLM_MATE_GDN_C1_PARTIAL_KKT=0`：回滚 generic varlen KKT。
- `XLLM_PIECEWISE_CAPTURE_FA3=0`：回滚 split FA3 runner。
- `XLLM_PIECEWISE_CAPTURE_FA3_MAX_PADDING_TOKENS=N`：full capture padding 上限，默认32。

Build log：

`build_logs/build_safe_full_fa3_partial_kkt_20260802.log`

Binary SHA256：

`a76856f6e20b78bf9be7776f4b584edc505221dc98de8dcca2d418e8d8806ddc`

## 6. Go / No-Go

- GO：C1 fixed partial-tail KKT 默认启用。
- GO：padding≤32 的 C1 full-FA3 capture 默认启用。
- GO：adaptive piecewise 在生产 mixed fixed-20 上快于 pure eager。
- NO_GO：padding>32 的 full-FA3 capture。
- 未声称：B>1 / packed multi-sequence full-FA3 已验证。
