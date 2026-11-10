---
title: "DeepSeek-V4.1"
description: "DeepSeek-V4.1-Flash-1017 W8A8 在 Ascend 上的权重适配与准备指南"
---

<!-- Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0. -->

# DeepSeek-V4.1-Flash-1017 W8A8

本页介绍 Eco-Tech DeepSeek-V4.1-Flash-1017 W8A8 checkpoint 在 xLLM Ascend Python 执行器中的权重格式与准备流程。

DeepSeek-V4 与 V4.1 的模型结构、Engram 内存需求和草稿模型不同。不要直接复用 [DeepSeek-V4 部署示例](/zh/cookbook/autoregressive_models/deepseek/deepseek_v4/)的设备数量、并行配置或 MTP 导出方式。第 6 节提供已验证的单机最小生成配置，不代表生产部署拓扑或性能结论；其他环境的资源预算仍需独立评估。

## 1. 模型与环境

权重地址：
https://www.modelscope.cn/models/Eco-Tech/DeepSeek-V4.1-Flash-1017-w8a8

使用包含 1017 适配的 xLLM 源码或安装包，并在同一 Ascend 环境中完成构建、准备及启动。NPU Python 模型执行使用 `--model_impl=python` 和 `--npu_kernel_backend=TORCH`。

本页面向 W8A8 导出，不代表官方 1017 FP8 checkpoint 的部署指南。

构建并安装当前 xLLM 自定义算子后，还需在启动环境中加载各 vendor 的 `bin/set_env.bash`，确保 `ASCEND_CUSTOM_OPP_PATH` 包含实际安装的算子目录。本次 CANN 9.1 验证使用 `custom_xllm_math` 与 `custom_transformer`；仅在磁盘上安装 vendor 并不等同于已加载其环境。缺少该设置可能在真实前向时触发 `Cannot find kernel by name SmlaVaMetadata`，不要通过替换 dtype 或跳过 Attention 绕过。

## 2. 混合精度权重

| 模块 | 1017 导出格式 | xLLM 加载方式 |
|---|---|---|
| Attention `wq_a` / `wkv` | BF16 | 普通浮点 Linear |
| Attention `wq_b` 与量化 FFN | W8A8 | 动态 W8A8 Linear / MoE |
| Engram embedding 表 | INT8 + UE8M0 group-32 scale | 保持 INT8 表，加载时分块解码 scale 为 FP32 |

xLLM 从 `quant_model_description.json` 为主干和 DSpark 各层选择投影精度。请保留该文件，不要通过删除量化描述、伪造 scale 或把 BF16 权重直接转成 INT8 来绕过检查。

UE8M0 scale 的 safetensors dtype 为 `F8_E8M0`。加载时只解码当前 TP rank 拥有的行，运行期仍使用 FP32 scale；将原始字节直接转换成浮点数不是正确的 UE8M0 解码。旧版全 W8A8 Attention 与 FP32 Engram scale 路径保持兼容。

## 3. 准备 Engram hash-state

Engram 推理需要 tokenizer-derived `engram_hash_state.bin`。若权重包未提供，先使用同一 checkpoint 的 tokenizer 生成。该步骤需要 `tokenizers` 和 `numpy`。

```bash
MODEL_PATH=/path/to/DeepSeek-V4.1-Flash-1017-w8a8
python - "$MODEL_PATH" <<'PY'
import sys
from xllm.python.models.deepseek_v41_engram import build_engram_hash_state

print(build_engram_hash_state(sys.argv[1]))
PY
```

生成器保留特殊 token，并按官方归一化规则移除全部 Unicode Mark 类字符。不要修改 `engram_compressed_vocab_size` 来绕过数量校验，否则会重哈希到错误的表行。1017 的压缩词表数量为 `99092`。

只读模型目录可使用可写模型视图：链接原始 safetensors 文件及 `optional/`，保留 config、tokenizer、量化描述和权重索引文件，再将 sidecar 写到视图中。启动时将 `--model` 指向该视图，无需复制或改写大权重文件。

## 4. Engram 内存与卸载

NPU Engram 的 CPU 卸载与 DP 共享默认均关闭。运行期 scale 为 FP32，因此 UE8M0 源格式不会自动缩小运行期表的内存预算。请独立评估 TP、DP、EP、HBM 与主机内存，不要套用 V4 的资源配置。

主机内存充足时，可通过 `--engram_config` 显式选择 CPU 卸载而不启用 DP 共享：

```json
{"cpu_offload":true,"dp_shared_memory":false}
```

DP 共享还需要 Gloo 通信和足够的共享内存，并须验证大表加载的耗时与超时条件。详细配置及约束见 [Engram 主机侧卸载](/zh/features/engram_host_offload/)。上述配置不是完整启动命令。

## 5. DSpark 草稿模型

1017 的主干与 DSpark Attention 都需要按 checkpoint 描述选择混合精度。使用 DSpark 时，target 与 draft 指向同一份已准备的 checkpoint 或模型视图；不要套用 V4 旧 MTP 路径的 `export_mtp.py`。

DSpark 的完整启动与生成需单独验证，不能用主干权重加载或单算子通过代替端到端结论。相关选项见 [命令行参考](/zh/cli_reference/)。

## 6. 已验证的最小生成配置

2026-11-09 在单台 Ascend A3（16 个逻辑 NPU，每个 64 GiB HBM；主机内存约 3 TiB）、CANN 9.1、Torch / torch_npu 2.10 环境中，分别完成普通生成与 DSpark 端到端生成冒烟验证。两种模式的真实 `/v1/chat/completions` 请求均返回 `Paris`、`2` 以及 `1, 2, 3, 4, 5, 6, 7, 8, 9, 10`，completion token 数分别为 2、2、29，均正常结束。DSpark 的草稿与接受 token 指标非零，确认实际执行了草稿与验证。

验证使用逐块校验一致的本地权重缓存与已准备的 Engram sidecar，没有改写原始 checkpoint。网络文件系统的大表 mmap 读取可能显著拉长启动时间；请在实际存储上评估 I/O 或使用容量充足的本地缓存。

下面是该单机验证的最小配置，解析为 TP8 / Attention DP2 / MoE EP16；`MODEL_PATH` 也可指向已准备的可写模型视图。先加载 Ascend 和自定义 vendor 环境，再启动：

```bash
MODEL_PATH=/path/to/DeepSeek-V4.1-Flash-1017-w8a8
XLLM_BIN=/path/to/xllm
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

python -m xllm.launch_server serve \
  --binary-path "$XLLM_BIN" --nnodes=16 --port=29381 \
  --model="$MODEL_PATH" --model_id=deepseek-v41-1017 \
  --host=127.0.0.1 --master_node_addr=127.0.0.1:29481 \
  --backend=llm --model_impl=python --npu_kernel_backend=TORCH \
  --dp_size=2 --ep_size=16 --max_memory_utilization=0.90 \
  --max_cache_size=1073741824 --block_size=128 \
  --max_tokens_per_batch=256 --max_seqs_per_batch=1 \
  --enable_prefix_cache=false --enable_graph=false \
  --enable_schedule_overlap=false --enable_fused_mc2=0 \
  --engram_config='{"cpu_offload":true,"dp_shared_memory":false}'
```

验证 DSpark 时，在上述启动命令末尾追加：

```bash
--draft_model="$MODEL_PATH" --speculative_algorithm=DSpark --num_speculative_tokens=5
```

请求设置 `temperature=0`、`reasoning_effort="none"`。DSpark 验证走兼容 SAS 路径，未开启 native SAS。此次验证关闭了图模式、调度重叠及 DP 共享表，不覆盖这些模式，也不构成并发、长上下文、精度评测、多模态或性能承诺。
