# V4.1：保留 Python 模型入口，逐项复用已有 MUSA 算子

> 本页记录共享 SwiGLU 提交 `34815f6a` 的阶段状态；后续三项 mHC 接入、1017 双机测试与新发现见 [后续实验记录](../20261110_deepseek_v41_mhc_1p1d/README.md)。

## 1. 本轮边界与基线

按照 11 月 10 日确认的方向，继续在共同仓库 `fay85/xllm-musa` 的个人分支 `xiaopangDeepseeekV41Flash` 上开发，暂不重写整套 C++ 模型。起点为 `4ade93777acdcd2d8271841b713809513c40886c`，其共同基线为 `1103-merged-dev` 的 `0cfd82700e164e8aa7951064b652925dd6b24d18`。只提交个人分支，不发 MR/PR。

参考家明的 `feat/dsv4-pcp-mate027-verified-20261029`，固定提交 [`f8dd0663279e9085365ade309d9823b7b438c4e6`](https://github.com/fay85/xllm-musa/commit/f8dd0663279e9085365ade309d9823b7b438c4e6)，再核对这些实现已进入 1103 基线后的具体文件。参考对象是这次固定下来的代码，不以分支名推断今天的最新状态。

Python 仍组织 V4.1 forward；C++ 仍承担 xLLM 框架执行和通信。可以从 Python 调用已有的编译 GPU 算子，无须先把整个模型改写为 C++。本次第一项复用的是 **xLLM 内已有的 MUSA 自定义 kernel**，不是新开发一套 SwiGLU，也不是宣称已经接入完整 MATE MegaMoE。

## 2. 为什么先做共享专家的 clamped SwiGLU

当前共享专家的流程是：gate/up 两个投影 → 转 FP32 → gate 只夹上界、up 夹上下界 → SiLU 和逐元素乘法 → 转 BF16 → down 投影。输入宽度、数值公式和调用位置都容易单独核对，而且不涉及专家分发、P/D 状态或缓存格式。

家明版本的 `xllm/core/kernels/musa/deepseek_v4_wo_a_m1.cu` 已有 `deepseek_v4_shared_swiglu`，V4 的 `deepseek_v4_moe.cpp` 已实际调用它。我们复用这个一般形状的激活 kernel，**不套用同文件内有特殊维度限制的整套 M=1 GEMV**。

路由专家暂不一起替换。V4.1 的路由权重必须乘在 FP32 激活上，然后才转 BF16；把共享专家的 BF16 输出拿去再乘路由权重，会改变舍入顺序。新增测试明确覆盖了这个区别。

本次再次核对构造和权重加载后，需要特别区分：当前 V4.1 Python 路径的共享专家是每个 rank 保存完整 `I=2304` 的 `nn.Linear`，不是已按 TP8 切成 `I=288`。Attention 的 TP8 不代表所有子模块都是 TP8。基准中 2304 是当前共享专家宽度，288 是便于后续分片评估的补充维度，不能将后者的数据当作当前部署实测。

## 3. 改动在哪几层

| 层次 | 文件 | 本次改动与原因 |
|---|---|---|
| 服务、调度与 P/D | 无 | 本项不改变服务协议、请求批处理或通信 |
| 执行层的 Python/C++ 绑定 | `xllm/core/runtime/py_musa_kernel_bindings.h`、`py_runtime_module.cpp` | 在真实 `xllm_runtime` 注册 `musa_shared_swiglu`；检查设备、BF16、形状、连续内存与 clamp；分配输出后调用已有 kernel。需要重编 xLLM |
| 模型层 | `xllm/python/models/deepseek_v41.py` | dense/shared MLP 与 MoE 的 shared 分支在构造时选择实现；路由专家仍沿用原有 FP32 乘权重顺序 |
| Python 后端适配 | `xllm/python/kernels_musa/shared_swiglu.py` | 显式选择 reference 或 v4_fused；拼接 gate/up 后调用编译绑定。不会把 GPU 计算搬到 CPU，也不会在运行失败后悄悄回退 |
| GPU kernel | `deepseek_v4_wo_a_m1.cu` | 复用原 kernel；修正 clamp 对 NaN 的传播，有限值公式不变 |
| 验证入口 | `musa_python_kernel_test_module.cpp`、`validate_shared_swiglu.py`、`validate_csa2_fixture.py` | 独立测试模块复用同一个绑定头文件和同一个 kernel，避免用 mock 冒充 GPU 验证；小模型可保存 hidden/logits 进行 A/B |

环境变量为 `XLLM_DSV41_MUSA_SHARED_SWIGLU=reference|v4_fused`，默认 `reference`，在构造模块时读取。选择 `v4_fused` 但使用旧二进制时，直接报告缺少 `xllm_runtime.musa_shared_swiglu`，要求重编；不以 Python 文件已更新来冒充新算子已生效。此开关只作用于 MUSA 分支，NPU/CUDA 不因此改变选择。

融合入口目前接受 BF16 的二维 gate/up，clamp 必须有限且在 `(0, 1000000)` 内。空 token 批次返回相应空张量。拼接会产生一次 `cat`，输出还有一次分配；这些都计入性能测试，不能把只测裸 kernel 的时间当成整条调用的收益。这里是 eager 推理接口，尚不承诺 graph capture 或训练反向传播支持。

## 4. 本轮发现与验证标准

初版直接复用 V4 kernel 时，有限数值的 10 组测试通过，但 NaN/Inf 边界测试失败：169 个组合中有 24 个不符合参考结果。原因是 `fminf/fmaxf` 会在一侧为 NaN 时选择另一侧的有限值，而 `torch.clamp` 保留 NaN。改成保留 NaN 的比较选择，防止上游异常被激活函数掩盖。失败日志保留，修复后重新编译并验证。

数值阈值在测试前固定为激活输出 `rtol=1/128, atol=1e-7`，同时要求 NaN/Inf 分类一致；记录实际最大绝对误差和不一致元素数。测试包括：T=0/1/17/256/2048、I=288/2304、非连续输入、clamp 边界、非默认 MUSA stream，以及含原 packed FP4 路由专家的完整小 MoE 调用。

验证分层：CPU 逻辑与路由舍入回归 → GPU 实际绑定与算子 → 随机小模型 hidden/logits 和 prefill/decode → 官方权重基础问答 → 真实双机 1P1D 两组测速。前一层通过不等于后一层已验收。

微基准记录同步后的 wall time 和 MUSA event time，每组 10 次暖机、5 组样本、每组 50 次调用；同时测 reference、含 cat/分配/绑定的 fused、预先拼接输入后的 native 入口。它衡量激活阶段，不是 TTFT、TPOT 或整模型 token/s。

补充测了包含 gate、up、down 三个 BF16 GEMM 的完整共享 MLP，hidden=5120、固定相同的随机权重。该组也是 10 次暖机、5 个样本，每个样本 10 次调用。最终结果显示小输入获益，大输入存在退化，因此仍默认 reference，不全局打开 fused。

实际运行结果、构建版本和未完成事项见同目录的 `验证结果.md`。不得将“已编译”“单算子通过”写成“1017 完整适配完成”。

## 5. 对照 V4 后的模块清单与下一步顺序

| 优先级 | 模块 | V4 可借鉴的具体实现 | V4.1 的差异与可落地做法 | 状态 |
|---|---|---|---|---|
| P0 | shared clamped SwiGLU | `deepseek_v4_wo_a_m1.cu`，V4 shared expert 的调用点 | 复用编译 kernel，保留 gate/up clamp 与 FP32 中间值、BF16 输出；先独立验证 | 本次实现 |
| P1 | mHC 独立的 collapse、post、Sinkhorn metadata | `deepseek_v4_dsa_cache_ops.cu` 的 `hc_pre_combine`、`hc_post`、`hc_pre_metadata`，以及 `deepseek_v4_decoder_layer.cpp` 的组织方式 | 可以逐个核对接口并绑定；必须保留 V4.1 shifted pre_mix，不能直接把整个 V4 `hc_pre_decode` 套进来。验证 comb 转置方向、FP32 累加、epsilon、Sinkhorn 轮数和跨子层系数来源 | 已定位候选；尚未接入/验收 |
| P1 | RMSNorm、RoPE 等小算子 | V4 attention 和已有 MATE norm/自定义 RoPE 调用 | V4.1 `_rms_norm` 当前要求 FP32；不能为了调用半精度 MATE 接口强制降精度。RoPE 要核对 interleaved、逆旋转和不同层的频率表后逐个替换 | 保持已有参考路径，先核对语义 |
| P1（收益大，但不属于最容易项） | FP4 专家 GEMM + block32 | V4 `deepseek_v4_moe.cpp` 的 packed weight、路由、grouped GEMM/GEMV、通信和 combine | V4 FP8 高性能专家路径不能直接视为 V4.1 FP4 支持。V4.1 要核对 E2M1 权重、E8M0 scale、FP8 激活每 32 元素的 scale、gate/up/down 布局及舍入。已有 MATE FP4 primitive 可复用，但 per-token A scale 不能直接替代 block32 | 原基线仍按活跃专家解码为 BF16；完整高性能路径未接入 |
| P2 | CSA2 attention/indexer/cache | V4 `deepseek_v4_attention.cpp` 的分页 metadata、稀疏 attention、Indexer、压缩与物理页映射 | V4.1 有 Full/Reindex/Reuse、跨层 KV/Top-K 复用、不同量化格式。先做无状态单算子，再改缓存写读；不能只比较输出 shape | 现有 GPU reference 可运行；官方量化与融合待补 |
| P2 | Engram | V4 没有可直接替换的同结构完整实现 | 复用我方已校对的 tokenizer hash、状态及表加载；再看 lookup、gating 和多卡数据移动。不能把 CPU 离线构造 hash 混同于在线 GPU 算子回退 | 已有基础实现；优化待做 |
| P3 | CED 的 Decoder Prefill 尾部 SWA replay | 可参考 V4 的 P/D 缓存管理机制 | V4.1 的 replay 和跨层依赖要单独组织；需验证 P 侧输出、传输状态及 D 侧续算一致 | 原测试未开启，不能声称完整支持 |
| 后续独立任务 | DSpark/MTP、graph、图像输入 | V4/飞狐的分离调度、推测解码与 graph 工作 | 先完成当前 eager 文本算子接入，再做跨 token 状态、缓存和 graph 地址稳定性验证 | 不属于本次小算子验收 |

“量化合同”在这里指官方权重与参考推理实现对 dtype、分组、scale 编码、舍入与计算顺序的约定；不需要领导临时发明一个精度要求，也不能由 MATE 方便支持什么就反过来定义模型语义。应将官方约定写成输入/输出回归，再问 MATE 是否已有相符接口。没有相符接口时，明确提出缺口，而不是偷偷用另一种量化替代。

给 MATE 同事的具体问题可收敛为：现有版本是否提供 **A=FP8 E4M3、沿 K 每 32 元素 E8M0 scale，B=packed E2M1 FP4、沿 K 每 32 元素 E8M0 scale** 的专家 GEMM，并覆盖 V4.1 的 gate/up、down、prefill 与小 M decode 形状？如果有，请提供函数签名、支持 shape/stride、scale 排布、示例与对应 wheel/source 版本。mHC 则问是否能独立消费外部 pre_mix；先复用独立原语不必等完整融合接口。

## 6. 后续验收

保留 reference，按一个改动一轮 A/B 推进。优先完成共享激活的真实权重 hidden/logits 对照与基础题，合格后在空闲的两台 S5000 上测 1P1D、每台 TP8、eager、并发 1，256/256 与 2048/256 各 1 次暖机加 5 次正式请求。输入 token、输出长度、模型版本、MATE/Torch/驱动、分片与通信配置保持一致。

局部激活阶段加速不能推出整体相同倍数加速。尤其当前逐专家解量化、Python 循环、GEMM、CSA2 与通信都未因本项改变；整体收益必须实测。当前首要目标是已有算子的正确接入，之后再用 profiler 判断剩余时间花在哪里。
