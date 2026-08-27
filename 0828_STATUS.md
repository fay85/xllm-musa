# 2026-08-28 Qwen3-0.6B MUSA software-beam 状态

> 本文是当前 handoff 的权威入口，取代 `0827_STATUS.md` 中已经过时的“CPU fallback 当前状态”描述。
> `0827_STATUS.md` 保留完整诊断历史；本文聚焦已经落地的修复、验证结果、本地 Git 快照和仍未关闭的问题。

## 0. 当前结论

Qwen3-0.6B software-beam 的 graph-on 故障已经通过两层修复解决：

1. software-beam TopK 不再使用可能被 graph/caching allocator 复用的普通 Torch 输出与 workspace，改为 current MUSA stream 上的 muDNN C API，以及进程生命周期 grow-only MUSA buffers；
2. piecewise prefill 的 padding cache slot 不再复制最后一个真实 token 的 slot，改成 `-1`，由 paged-cache kernel 跳过 padding 写入。

`beam_width=128` 已完成 graph-on 准入：

```text
C=1: 1000/1000，unstable=0
C=4: 2000/2000，unstable=0
C-Eval graph-on/off: 都是 431/1346（32.0208%）
没有新 .mudmp
```

当前代码中没有 CPU fallback。`musa_topk.cpp` 中的 D2H 是越界诊断检查，不参与结果计算；但是同步和 D2H 检查仍有明显性能代价，后续优化需要在稳定性门槛之后单独进行。

`beam_width=512` 尚未准入：graph-on 能完成 1000/1000，但有 3 个重复 prompt 的单次输出签名 outlier；graph-off 在首批请求中被 `MUSA_TOPK_INDEX_CORRUPTION` 检查捕获。因此不能写成“beam=512 没问题”。

## 1. 权威环境

```text
SSH:             ssh dev92
host source:     /data/feihu/xllm-git-master-python
container:       xllm-musa2.9.1-sdk5.1-dev
container source:/workspace/xllm-git-master-python
model:           /data/nfs_shared/models/Qwen3-0.6B
MUSA SDK:        5.1.0
torch:           2.9.1
torch_musa:      2.9.1+a18d871
mate:            0.2.6
tvm_ffi:         0.1.11.post1+musa.1
tilelang_musa:   0.1.12+musa.2
```

构建、server、correctness 和 benchmark 必须在容器内执行。不要在 host `/data/feihu` 直接构建，不要运行 bare Ninja。

必须先读：

```text
/data/feihu/xllm-git-master-python/AGENTS.md
/data/feihu/xllm-git-master-python/.agents/skills/musa-container-build/SKILL.md
/data/feihu/xllm-git-master-python/.agents/skills/add-unit-test/SKILL.md
```

修改或 review `xllm/` 前还必须完整阅读本地及 upstream canonical style 文件；离线时不能伪称已完成 upstream 对拍。

## 2. 已落地的 bugfix

### 2.1 MUSA software-beam TopK wrapper

文件：

```text
xllm/core/kernels/musa/musa_topk.cpp
xllm/core/kernels/musa/musa_ops_api.h
xllm/core/kernels/musa/CMakeLists.txt
xllm/core/framework/sampling/sampler.cpp
```

关键实现：

- 仅服务 `params.max_top_logprobs > 0` 的 MUSA software-beam 路径；
- input 为 2-D FP32 `[rows, vocab]`；
- muDNN handle 每次绑定 `c10::musa::getCurrentMUSAStream().stream()`；
- input/values/indices/workspace 使用 thread-local grow-only `musaMalloc` buffer；
- buffer 扩容时不立即释放旧地址，避免 captured graph 或 in-flight kernel 仍引用旧地址；
- TopK 前后和输出 copy 后执行 device synchronization；
- 输出 indices 拷回 host 做 `[0, vocab)` 范围检查，异常时 `LOG(FATAL) MUSA_TOPK_INDEX_CORRUPTION`；
- `sampler.cpp` 直接调用 `xllm::kernel::musa::topk`，没有 large-K CPU reference fallback；
- 显式链接版本化的 `libmudnn.so.3` C API，同时保留原有 `libmudnncxx.so.3`。

当前源码中不存在：

```text
kMusaMaxStableTopK
musa_safe_topk
cpu_input
```

### 2.2 piecewise prefill padding KV 污染修复

文件：

```text
xllm/core/runtime/musa/musa_graph_executor_impl.cpp
```

旧逻辑把 pad query 的 `new_cache_slots` 设为最后一个真实 token 的 slot。pad query 位置不同，其 K/V 不等于最后一个真实 token，因此会覆盖真实 KV，并导致非 bucket-aligned ISL 每次输出变化。

新逻辑：

```text
piecewise prefill pad slot = -1
reshape_paged_cache_kernel: slot_id < 0 时直接 return
```

该修改只作用于 piecewise prefill padding 分支；decode/packed 的既有 `-1` 语义没有扩大。

### 2.3 Mate 0.2.6 runtime/build 对齐

文件：

```text
_build_cuda_graph_musa.sh
run_xllm_musa.sh
```

当前 launcher/build script 明确校验：

```text
mate==0.2.6
apache-tvm-ffi==0.1.11.post1+musa.1
tilelang-musa==0.1.12+musa.2
```

Qwen3 D128/GQA2 的 Mate 0.2.6 metadata artifacts 已补齐 `x1...x17`，因此 `MAX_SEQS_PER_BATCH=512` 的 C=4 beam 测试可以通过 preflight。

## 3. 当前源码和二进制身份

准入二进制：

```text
/data/feihu/xllm-git-master-python/build/lib.linux-x86_64-cpython-310/xllm/xllm
SHA256: 7186c21ceceea68a9174176f743fd4dfde395f3297aea6b1f8b8ba0b629c4667
build log: build_logs/build_cuda_graph_musa_20260827_143055.log
```

关键源码 SHA256：

```text
f82ccd006e4470a230bc6639fd1c2be2badf8623ff0589cfacae90529b8c4df4  xllm/core/framework/sampling/sampler.cpp
0ec92d2db55b8834f4bb3af411f24ebc74a789dbb999b8e02bf62fd3637b6d3b  xllm/core/kernels/musa/musa_topk.cpp
f56991f23f8cdb57ded644058968912ebbcb05b152d75d01f4fff3a95dd6e4cc  xllm/core/kernels/musa/musa_ops_api.h
649b8416ab2b99dd9fc70b4d9bf1aa4f1e7acb2201df4f304e67b046d8e1b46e  xllm/core/kernels/musa/CMakeLists.txt
39569c47a4c76b00d597c4454ba1dc3f0d041d006daaccc76c764fac69ca949b  xllm/core/runtime/musa/musa_graph_executor_impl.cpp
```

## 4. Correctness 与稳定性

### 4.1 standalone TopK regression

脚本：

```text
tests/python/test_mudnn_topk_regression.py
```

默认 Qwen3-0.6B software-beam shape：

```text
input: FP32 [128,151936]
k: 256
iterations: 1000
检查: index dtype/range、values==gather、CPU candidate set
结果: PASS
```

beam=512 对应 standalone shape 也通过：

```text
input: FP32 [512,151936]
k: 1024
iterations: 1000
结果: PASS（23.30 s）
```

这只能证明 standalone muDNN/TorchMUSA 不必然出错，不能替代集成测试。

### 4.2 普通 correctness_check.sh

当前 graph-on 二进制：

- Qwen3 `/no_think`、C=4：HTTP、非乱码、token、`391`、并发逐字一致全部 PASS；
- 原始 thinking prompt、C=1、`MAXTOK=512`：包含正确答案 `391`，脚本 PASS；
- 原始 thinking prompt、C=4：四路都包含正确答案，但完整 thinking 文本不逐字一致，因此旧脚本的 `concurrent_match_golden` 不适合作为 0.6B thinking-mode 门槛。

结果目录：

```text
/data/feihu/validation_qwen06_mate026_20260827/current_verify
```

### 4.3 beam=128 集成稳定性

固定协议：

```text
client:      /data/feihu/qwen06_errorfile_client.py
workload:    /data/feihu/validation_qwen06_errorfile_20260824/workload.jsonl
beam_width:  128
TopK k:      256
output_len:  3
graph:       on
piecewise:   on
ISL:         1...1024，32 个固定 prompt
seed:        1234
```

结果：

| Concurrency | Requests | Success | Unstable prompts | QPS | Mean | P99 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1000 | 1000/1000 | 0 | 11.0703 | 89.57 ms | 139.38 ms |
| 4 | 2000 | 2000/2000 | 0 | 13.0040 | 306.97 ms | 465.66 ms |

产物：

```text
/data/feihu/validation_qwen06_mate026_20260827/current_verify/beam_c1_1000.json
/data/feihu/validation_qwen06_mate026_20260827/current_verify/beam_bench_c4_2000.json
```

### 4.4 C-Eval graph-on/off 对拍

协议：52 subjects、1346 questions、5-shot、temperature=0、C=4。由于 Qwen3-0.6B 不遵守旧的 27B `max_tokens=2` answer-only 输出，协议校准为显式要求只输出选项，并接受开头的：

```text
A/B/C/D
答案：A/B/C/D
A. option text
```

不会从解释中猜测答案。

| 模式 | Correct | Accuracy | HTTP failures |
|---|---:|---:|---:|
| graph-off | 431/1346 | 32.0208% | 0 |
| graph-on | 431/1346 | 32.0208% | 0 |
| graph-on repeat | 431/1346 | 32.0208% | 0 |

两次 graph-on 全量运行的 1346 个 raw response、prediction、correctness 逐题完全一致。graph-on 与 graph-off 有 2/1346 选项差异，但一题由对变错、一题由无效变对，净精度差为 0。

产物：

```text
/data/feihu/validation_qwen06_mate026_20260827/current_verify/ceval_graph_off_v3
/data/feihu/validation_qwen06_mate026_20260827/current_verify/ceval_graph_on_v3
/data/feihu/validation_qwen06_mate026_20260827/current_verify/ceval_graph_on_repeat
```

32.0208% 是 0.6B 模型和当前适配协议的绝对能力分数；本轮主要使用其 graph-on/off 一致性作为修复精度门槛。

## 5. beam=512 尚未关闭

测试配置：

```text
beam_width: 512
TopK k:     1024
C:          1
output_len: 3
ISL:        1...1024
max_seqs:   512
```

graph-on 短测：

```text
200/200
errors=0
unstable_prompt_count=0
QPS=1.9536
mean=509.49 ms
p99=681.83 ms
```

graph-on 长测：

```text
1000/1000
errors=0
unstable_prompt_count=3
QPS=1.9646
mean=506.50 ms
p99=677.92 ms
```

3 个 outlier 分别是 ISL 1、2、32 的重复 prompt，每个只有一次签名偏离，其余 30/31 次一致。这不是请求结构失败，但不满足 `unstable_prompt_count=0` 的准入标准。

graph-off 对照在首批请求中被当前现场检查捕获：

```text
MUSA_TOPK_INDEX_CORRUPTION
flat_index=175104
row=171
column=0
token_index=3007281507778442268
topk_value=-0.0546234
vocab_size=151936
```

日志：

```text
/data/feihu/validation_qwen06_mate026_20260827/beam512_verify/graph_off_server/xllm_Qwen3-0.6B.log
```

同 shape standalone TorchMUSA 1000 次，以及 direct muDNN C API 的 direct/cached workspace 各 1000 次均通过。因此 beam=512 问题仍属于 xLLM 集成 wrapper/调度/相邻状态，不是简单的 muDNN `k=1024` 必现缺陷。

当前结论：

```text
beam=128: 准入
beam=512: 可返回完整结果，但稳定性未准入
```

## 6. 性能边界

当前 TopK wrapper 仍包含：

```text
musaDeviceSynchronize before TopK
musaDeviceSynchronize after TopK
musaDeviceSynchronize after output copies
indices D2H scan
```

它们是为定位和阻止错误传播保留的高开销保护。不要在没有重复稳定性证据时删除。后续性能优化应逐项 A/B，每一步都重新跑：

1. standalone TopK；
2. beam=128 C=1 两轮 1000；
3. beam=128 C=4 2000；
4. C-Eval graph-on/off；
5. 新 dump 检查。

## 7. dev92 本地 Git 版本控制

dev92 无需连接 GitHub/GitLab。当前修复已经保存到独立健康 worktree，并 push 到 dev92 本地 bare remote。

```text
worktree: /data/feihu/xllm-qwen3-beam-local
bare:     /data/feihu/local_git/xllm-git-master-python.git
branch:   bugfix/qwen3-beam-topk-local
base:     3f656fd87f9fdc63d88e53392a8123fa092d29a3
```

已有 commits：

```text
9f4348c4 build: align local musa runtime with mate 0.2.6.
e9696745 bugfix: stabilize qwen3 software beam graph execution.
ac7c2859 test: add qwen3 software beam topk regression.
3489d8b9 docs: document qwen3 software beam investigation status.
```

其他 agent 可以直接：

```bash
git clone /data/feihu/local_git/xllm-git-master-python.git
```

bare repo 的默认 HEAD 已指向 `bugfix/qwen3-beam-topk-local`。当前本地 worktree 与 bare branch 一致，`git fsck --full` 通过。

原权威工作树 `/data/feihu/xllm-git-master-python` 的 `.git/index` 仍损坏，未修复、未 reset。不要在原工作树执行会重建或删除 index 的操作。

原始复现脚本另有独立本地 Git：

```text
worktree: /data/feihu/xllm-debug-assets
bare:     /data/feihu/local_git/xllm-debug-assets.git
commit:   1c299de006c0199b40899442dd26b1552bf4202f
```

## 8. dump 状态

本轮 beam=128、C-Eval、bench 和 beam=512 检查都没有生成新的 `.mudmp`。最新文件仍为：

```text
/data/feihu/xllm-git-master-python/core_2026-08-26_23:11:20.164_worker38092_67441.mudmp
```

graph-off beam=512 由 host `LOG(FATAL)` 检查主动停止；`run_xllm_musa.sh` 当前设置 `ulimit -c 0`，因此该次没有新 dump。

## 9. 后续优先级

1. 保持 beam=128 当前准入实现，不恢复 CPU fallback；
2. beam=512 若不是近期交付要求，可以作为独立 follow-up，不阻塞 beam=128；
3. 若继续 beam=512，优先在 xLLM wrapper 中记录实际 rows/k/buffer 地址/stream，并复现 graph-off 的 `row=171` 损坏；
4. 在 beam=512 根因关闭前，不要把 `beam_width=512` 作为支持配置宣传；
5. 性能优化应先去掉或收窄 debug D2H/sync，但必须逐项验证，不能一次全部删除；
6. 所有新代码先提交到 dev92 本地 Git，再做破坏性 A/B，避免再次只保留在损坏 index 的工作树中。
