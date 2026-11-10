---
title: "Python 分阶段 Execution Pipeline"
sidebar:
  order: 6
---
<!--
Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

## 统一的执行契约

DSpark 将 backbone + LM head 组合为 `DSparkDecoderModel`，采样使用独立的 `DSparkProposalModel`。
两个执行阶段都是 `nn.Module`，沿用
`forward(input_ids, positions, input_embedding=None)` 签名。模型只消费所需输入，
不要求采样具备 Attention/KV 能力，也不把 logits 冒充成 token IDs。

Decoder Model 在一次 forward 中完成 backbone 和 LM head，固定返回 `(logits, hidden)`，
不因是否需要 confidence 改变输出结构。阶段间的 logits 通过既有 `input_embedding` 参数传给 proposal Model。
这个参数与 MTP 一样可以承载中间特征，不仅限于 token embedding。
`input_ids` 始终是真实 token IDs；proposal Model 从中取得每个请求的 anchor，不能把 logits 冒充 IDs。

模块专属辅助数据通过类型化 `ForwardContext.execution_contexts` 取得。只有请求需要 confidence 时，
pipeline 才把 backbone hidden 放进 `ProposalMetadata`；采样参数仍由 `ModelSamplingMetadataBuilder` 提供。
运行器负责建立和退出每次 forward 的 context，模型不会保存跨请求的当前输入。
层加载、层完成事件与 EPLB 输入只传给包含 backbone 的 Decoder 阶段，不重复传给 proposal。

拥有 metadata builder 不代表所有请求都必须能入图。模型可以提供 `can_execute_graph`，
图执行器综合模型能力与本批布局、容量判断，统一 `Runner` 消费其判断，在执行前选择 graph 或 eager。
capture/replay 失败直接报错，不在可能已有副作用后重跑 eager。

## 层次与所有权

```text
ModelExecutor
  └─ DSparkExecutionPipeline
       ├─ entry_runner    → Runner(backbone + LM head) → EagerRunner / 原有 DecodeAclGraphRunner 等后端
       └─ proposal_runner → Runner(proposal) → EagerRunner / AclGraphRunner
```

`model_executor/runner.py` 的 `Runner` 是单个 Model 的统一执行入口，组合而不是继承具体执行后端。
每个 `Runner` 绑定一个 Model 及其 metadata builders，持有 eager 和服务配置允许的 graph 后端。
其 `execute(ModelInput)` 负责消费图后端的 `can_execute`、选择和调用后端；执行模式、上次图 padding
行数和历史恢复回调也归本 Runner 所有，避免不同 Model 的状态相互覆盖。

`ExecutionPipeline` 在自己的构造过程中通过 `register_model(model, builders)` 创建对应的 `Runner`。
每个 Runner 在构造时就创建 eager 和服务配置允许的 graph 后端，绑定各后端的 metadata builders；
所有后端引用同一个 Model，不重新加载或复制权重。没有先构造空 Runner、再调用 `bind_backends` 的阶段，
Executor 也不再遍历 Pipeline 的 Runner 补装后端。

`RunnerConfig` 汇集已有的设备、模型配置、graph 开关和容量等初始化参数，不引入新用户开关。
Executor 保留当前单主模型的 Attention backend 准备，把它、RunnerConfig 和可选历史恢复回调交给 Pipeline。
Pipeline 只把配置传给 Runner，不解析后端选择规则；主模型 Runner 使用原有 decode 后端，
proposal 不携带 Attention backend，使用非 Attention 后端。执行环境兼容性和容量约束仍由后端负责。

构造后端不等于 capture 图。KV cache 仍在上游资源就绪后绑定；warmup、首次 capture 和 replay
仍按原来的执行时机发生。Runner 和后端不按 step 重建。本次不扩展多个独立 Attention 主模型的资源装配。

具体 pipeline 的 `execute` 只调用 `entry_runner.execute`、`proposal_runner.execute`，
决定计算顺序、传递数据及等待时机。Pipeline 不解释 graph 开关，
不创建 `EagerRunner` / `AclGraphRunner`，也不判断本次执行方式；公共 `_execute_model` 已移入 `Runner.execute`。
原来的 warmup、Engram history 恢复和模型层加载等待顺序保持不变，恢复回调只绑定到 entry Runner。

没有提供 pipeline 构建入口的模型使用 `SingleModelExecutionPipeline`，保留原有
Tensor/tuple 返回和 eager/graph 执行路径。通用 executor 不判断 DSpark 类型。

`models/dspark_model.py` 中的 `DSparkModel` 是组件和权重容器，没有整体 `forward`。
它保持 checkpoint、共享 LM head 和 C++ 辅助方法的访问路径。两个计算 Model 引用该容器的权重，
不把整个容器重新注册成自己的子模块；pipeline 持有容器，保证运行期间资源存活。
`pipelines/dspark_pipeline.py` 是唯一的 DSpark 编排入口，即使所有阶段均走 eager 也不例外。

## 编排与异步的封装边界

公共接口统一的是 `execute` 的输入输出契约，不是内部调度算法。DSpark 当前直接按顺序调用两个 Model；
其他具体 pipeline 可以在内部采用线程、Future 或多个 stream。公共层不要求统一任务图、执行句柄或
`ExecutionResult`，Executor 也不消费 Future。

Pipeline 返回普通 Tensor、tuple 或 `ModelOutput` 时，应已建立后续调用方 stream 所需的消费依赖，
不要求设备已经执行完，更不要求每次做全设备同步。内部异步任务的异常和资源生命周期由该 pipeline 负责，
不能把未处理的失败或仍在使用却已释放的缓冲区留给调用方。

模型允许直接表达自己的输入等待，例如从有类型的 context 取得 H2D ready event，在实际消费层等待。
产生及记录 event 的时机属于 pipeline，消费点的 wait 属于模型；不再额外包装成延迟输入访问器。
入图时仍须维护 event 与持久缓冲区的捕获绑定和 replay 约定。这里只记录后续复用的边界，本次不改造
Engram，也不在 draft 链路引入线程、Future 或新的 event。

## 输入资源与图执行

- Eager：runner 直接传递 forward 参数，builder 的 `build` 构造当前 Model 可见的辅助 metadata。
- Capture：runner 固定 forward 参数的地址，builder 的 `allocate_persistent` 分配自己的辅助资源。
- Replay：runner 更新参数缓冲区，builder 的 `update_persistent` 更新辅助资源；两者清理各自的 padding，
  不替换图捕获的 Tensor 地址。

`ForwardContext` 只是取数机制，不会让本批新 Tensor 自动替换图中的旧地址。
host 侧的 context 查询和分支不会在 replay 时重新执行；模型的图能力声明仍需保证相同图身份下的
控制流一致，或选择本阶段 eager。不能仅扩大一个 bool 的通过范围来复用不同计算结构的图。

没有 Attention 的 Model 仍复用 `EagerRunner`，但不执行 Attention prepare。
`AclGraphRunner` 固定普通 forward 参数，并复用上述 metadata 资源协议；它与保留的 `DecodeAclGraphRunner`
拥有相同的模型调用方式，后者继续处理 Attention/KV、DP 协议、decode 特殊布局和状态保护。
本次不将大型 decode 图实现重写成一个万能后端，两者共享底层 capture 入口。

无 Attention 的图执行器按请求数选 bucket，并保持每个请求的完整 query block。例如三个请求、每请求三个 token，
可填充为四个请求、十二个 token。图缓存同时区分 query width 和输入 Tensor 的非行维度、dtype。
该后端对空批、dummy、ragged 或超出容量的输入在执行前选择 eager，不引入新用户开关。

跨阶段张量复制到下一张图的持久输入，stream 等待保护生产和消费，记录源 Tensor 的 stream 使用。
Runner 根据输出契约去掉 padding：Tensor hidden/logits 按 token 数，proposal 按请求数。
Eager 的结构化输出也使用上游真实计数裁剪 dummy 请求，不把这项职责交给 Model 或 pipeline。

## DSpark 计算与回退

1. Decoder Model 在一次 forward 中调用 backbone 和原 LM head，让两者使用同一个 Runner 和捕获范围。
   它汇总两者的 graph 能力，只绑定 backbone 所需的 metadata builders，不因采样条件被阻止入图。
   pipeline 创建后的权重共享仍可见；backbone 的后端配置及运行时 hooks 保持可访问。
2. Proposal Model 消费 logits、anchor、采样 metadata 和按需使用的 hidden，保留逐位置 Markov 调分、
   原生采样、token 重映射及按需概率/confidence 输出。

| 条件 | Proposal 执行方式 |
| --- | --- |
| 无约束 greedy，且不需要概率或 confidence | 可使用自己的 ACL graph |
| penalty/filter、随机采样、概率或 confidence 输出 | 本阶段 eager |
| 关闭 ACL graph，或图后端不能处理本批布局/容量 | 本阶段 eager |

不为了统一图而强制计算随机采样、全量概率或 confidence。backbone 与 LM head 共同选择执行方式，
采样独立选择；复杂采样回退 eager 不影响前一个阶段的 graph。

## 验证边界

相关测试用例包含构造结束即完成后端关联、共享模型和 builders、Runner 后端选择及状态隔离、
Pipeline 只编排统一入口、统一 forward/context、显式中间参数、
持久输入地址、padding 更新、阶段独立回退与恢复、Markov 反馈、概率/confidence、权重共享、dummy 裁剪和
executor 组装。两个 Runner 的用例还覆盖 backbone 与 LM head 共用一次捕获、采样回退及恢复、
固定的 logits/hidden 输出以及组合阶段的 padding 裁剪。
CPU 模拟图只能验证编排和缓冲逻辑，不等价于真实 NPU capture/replay。

非 ACL 平台的 proposal 阶段先走 eager，Decoder 阶段沿用原有模型后端。涉及 collective 的阶段仍要求通信域内
一致的输入布局和调用顺序；Decoder Model 通过已有 graph 能力接口参与准入协调，不改变共享布局和 bucket 规则。
真实 NPU、TP 和性能仍需验证。分阶段会增加 graph launch 与中间张量搬运，不能据此直接声称 TPOT 改善。

## 设计回顾与职责总结

当前设计的核心是：

> **Model 定义“算什么”，Runner 决定“怎么执行”，ExecutionPipeline 决定“如何组合”，
> Executor 负责对接服务执行入口。**

这次调整不是单纯拆文件，而是把原来混在一起的几种职责分开。

### 1. 执行层次

```text
Executor
  └─ ExecutionPipeline
       ├─ Runner(Model A)
       │    ├─ EagerRunner ─┐
       │    └─ GraphRunner ┴─→ 同一个 Model A
       └─ Runner(Model B)
            ├─ EagerRunner ─┐
            └─ GraphRunner ┴─→ 同一个 Model B
```

图中的 GraphRunner 是图后端的泛称，实际使用既有的 `DecodeAclGraphRunner`、`AclGraphRunner` 等实现，
并且只在服务配置允许时构造。它不表示本次请求一定执行 graph。

| 层次 | 核心职责 | 不承担的职责 |
| --- | --- | --- |
| Executor | 准备运行环境、输入和服务边界信息，调用 Pipeline | 不编排内部阶段，不选择每个 batch 的执行后端 |
| ExecutionPipeline | 定义有哪些阶段，以及数据流、执行顺序和异步依赖 | 不判断各阶段走 eager 还是 graph |
| Runner | 绑定一个 Model，构造后端，并选择本次执行方式 | 不知道上下游有哪些阶段 |
| 具体执行后端 | eager 执行，或 capture/replay、持久缓冲区及 padding 处理 | 不定义业务编排 |
| Model | 实现一个阶段的计算逻辑，声明自身的 graph 能力 | 不选择执行后端 |

**Runner 组合具体后端，具体后端共享同一个 Model，不复制或重新加载权重。**

### 2. 构造与资源持有

```text
Executor 准备配置与依赖
  → 创建具体 Pipeline
    → Pipeline 构造所需 Runner
      → Runner 构造自己的 eager/graph 后端并完成关联
```

- **具体 Pipeline 知道有哪些 Model**，Executor 不需要提前知道数量。
- Pipeline 构造结束时，Runner 及其后端已经完整关联，不再由 Executor 二次补装。
- 固定组件还是可替换组件，只影响构造来源，不改变执行抽象。
- `DSparkModel` 保留组件、权重及共享权重的资源归属；各阶段 Model 引用这些资源，
  Pipeline 持有资源容器，保证其生命周期。

这里的 Model 数量指独立执行阶段的数量，不是权重加载次数，也不是模型树中所有 `nn.Module` 的数量。
不需要通过扫描模型树来推断应该创建多少个 Runner。

“构造完成”不等于设备资源全部就绪：**KV cache 仍可在上游准备好后绑定，
graph 仍按原有时机首次 capture。**

### 3. DSpark 的具体落地

```text
DSparkExecutionPipeline
  ├─ entry_runner    → Decoder Model (backbone + LM head) → (logits, hidden)
  └─ proposal_runner → Proposal Model → ModelOutput
```

两个阶段沿用统一的 `forward` 协议，没有再区分 Model、Unit，也没有另造两套 Runner 接口。

- 主要中间 Tensor 通过 `input_embedding` 显式传递。
- 模块专属辅助数据通过类型化的 `ForwardContext` 获取。
- proposal 不需要 Attention，就不为它伪造 Attention/KV 资源。

**阶段按执行特性划分，不按模块数量机械拆分。** backbone 和 LM head 作为一个执行单元共同捕获，
采样作为另一个执行单元独立回退。复杂采样需要 eager 时，前一个阶段仍可走 graph。
这是开发时确定组合边界、运行时选择执行方式，不是按 batch 动态合图或拆图。

### 4. Graph 与 metadata 的语义

之前“Model 必须无条件能入图”的假说已经放弃。现在的约定是：

- Model 通过 `GraphExecutionSupport.can_execute_graph(inputs: ModelInput)` 声明本轮计算和 metadata 的 graph 条件。
- 模块通过自己的 `can_execute_graph` 接收所需的运行时信息；Model 显式汇总直接使用的模块，再叠加组合条件。
- 图后端检查自身的输入布局、设备、容量和执行上下文，再消费 Model 的汇总结果；不越过 Model 查询 builder。
- Runner 在执行前选择后端，不在 graph 已产生副作用后捕获异常、重跑 eager。

因此，**不再为了保持一张万能图，强迫简单请求执行所有采样计算。**

Metadata builder 的三个核心接口仍然成立：

| 接口 | 职责 |
| --- | --- |
| `build` | 构造当前输入对应的 metadata |
| `allocate_persistent` | 分配模块拥有的持久资源 |
| `update_persistent` | 更新持久资源的内容 |

**拥有 builder 表示资源管理有明确归属，不代表所有输入都天然符合 graph 条件。**
后端负责 forward 参数缓冲区及输出裁剪，builder 负责自己的辅助资源。

#### 准入判断的调用链

```text
Executor.execute(step)
  └─ ExecutionPipeline.execute(step)
       └─ 各阶段的 Runner.execute(stage_input)
            ├─ 未启用 graph → eager.execute(...)
            └─ graph_backend.can_execute(stage_input)
                 ├─ 后端条件：布局、设备、bucket、容量、执行上下文
                 ├─ Model.can_execute_graph(stage_input)
                 │    └─ 模块.can_execute_graph(本模块需要的运行时信息)
                 └─ 必要的跨 rank 准入协调
            ├─ True  → graph.execute(...)
            └─ False → eager.execute(...)
```

**请求信息一路向下传递，能力判断一路向上返回；最终选择执行方式的只有 Runner。**
Pipeline 不重新判断 graph 条件，后端也不在已开始执行后重新做 Model 准入或失败重试。

`ModelInput` 定义在 `model_executor/model_input.py`，是阶段输入的数据契约，不是新的执行层。
Runner、图后端准入、Model 准入共享它，包含 Tensor 参数、Attention metadata、`InputBatch`
及已有执行上下文。模型需要采样信息时读取 `inputs.input_batch.sampling`，再将这部分信息传给
采样 builder；模块不必为了接收采样参数而依赖整个执行层。backend 的计算入口和 Model 的
`forward` 参数形式保持不变，由 Runner 在调用计算入口时解包。

这个接口必须是无副作用的准入检查：`False` 表示本次应选择已支持的 eager 路径，
`True` 仅表示计算条件成立，不承诺后端一定有合适的 bucket 或缓冲区。
不自动递归扫描整个 `nn.Module` 树；组合关系由拥有该计算路径的 Model/模块显式表达。

当前先在 DSpark draft pipeline 接通：

- Decoder Model 汇总 backbone 与共享 LM head 的能力，不汇总采样限制。
- DSpark proposal Model 调用 `ModelSamplingMetadataBuilder.can_execute_graph(sampling)`，并汇总 Markov head 的能力。
  只有无约束 greedy、无需 probabilities/confidence 时采样阶段入图；其他采样仍 eager，
  不连带否决 backbone 和 logits。Model 与后端持有同一个采样 builder 实例，持久资源校验复用其
  `can_execute_graph`，避免维护两份规则。这个改动不修改采样数值逻辑，也不扩大采样入图范围。

本次不迁移 target Model，也不改造 V4.1 target/draft 共用的 MoE 和 decoder layer。
V4.1 现有的 MoE 量化模式守卫、CSA2 启动校验继续保留，待后续专门迁移时再收敛归属。

`GraphExecutionSupport` 是结构化接口，没有实现时公共查询默认返回 `True`，表示**没有额外计算约束**，
不是绕过后端检查，也不是框架替模块证明了图安全。与请求无关的禁止条件可以直接返回 `False`；
有条件支持的模块根据本轮输入判断；普通模块无需添加恒为 `True` 的样板方法。
不自动扫描子模块：Model 必须显式汇总自己使用的条件模块，新增动态计算路径必须相应维护声明。
已有普通 Model 因此保留原有 backend-only 准入，避免一次迁移所有模型。builder 只被后端调用三个资源接口，
错误的持久化输入仍报错，不将资源准备失败吞成 eager fallback。

ACL 和 CUDA 的 backbone 图后端都消费 Model 准入；multimodal 以及 CUDA 下的 EPLB 限制也收回后端，
Runner 不再在 `can_execute` 之外维护另一组否决条件。对实现了 `GraphExecutionSupport` 的 backbone，
`dp_size > 1` 时通过 `dp_all_true` 汇总本 rank 的完整准入结果，所有 rank 同意才执行图；本 rank 拒绝也必须参与，
不能提前返回让其他 rank 等不到票。全局 token counts 仍负责共享 bucket 的选择，不能代替动态能力投票。
只有接入新接口的 DP backbone 增加这次准入 collective；未迁移的旧模型沿用原有共享布局准入，
不因这次接口改造额外增加投票。proposal 的 `AclGraphRunner` 默认仍只做本地准入；DFlash2 sampling
通过 Runner 显式启用 DP 协调，见第 7 节。

### 5. 异步编排的封装

公共接口不强制内部串行，也不要求暴露 Future 或统一任务图。

外部仍调用普通的 `pipeline.execute(...)`；具体 Pipeline 负责建立必要的消费依赖，
并管理内部异步工作的错误和资源生命周期。返回不要求 GPU/NPU 已经完成，
但不能让调用方拿到缺少必要依赖的结果。

对后续 DSV4.1，适配方向是：

- 具体 Pipeline 携带 Engram 与 DSV4.1 的组合语义。
- Pipeline 安排准备工作和 H2D。
- target 从 context 获取 ready event，在实际消费层等待。
- 不需要为了表达依赖而拆开 target，也不要求把固定组件做成可替换插件。

**这是已确定的设计方向，不是本轮已经完成的 Engram 改造。**

### 6. 当前边界与取舍

- 当前仍按**一个需要完整 Attention/KV 资源的主模型**落地，
  不提前扩展多个独立主模型的资源装配。
- Executor 仍处理 graph 相关的服务初始化信息和原有 KV 接线，因此不是源码层面完全与 graph 无关；
  但后端构造和逐 batch 选择已经下放。
- 分阶段能隔离 fallback，但也可能增加 graph launch 和中间 Tensor 搬运，
  **性能收益必须实测，不能仅凭架构判断。**
- CPU 单测与真实 NPU 验证是不同证据；本次阶段合并不据此承诺 TPOT 收益。

总体原则是：**统一执行入口，但不强求统一执行方式；把业务差异留在具体 Pipeline，
把执行差异留在 Runner，把资源管理留给实际拥有资源的模块。**

### 7. DFlash2 的逐 step 编排

DFlash2 与 DSpark 采用同一种跨 step 生命周期，而不是在上一轮预测下一批请求并缓存完整 draft：

- 第 `n-1` 轮完成 target verify 后，提交 accepted tokens 的 D2H，并在本轮提交 context KV 写入。
  异步输出携带覆盖 context 写入的完成事件，资源和块的释放必须等待该事件；最后一轮也不例外。
- 第 `n` 轮拿到实际 batch 后，再决定是否 prelaunch。用请求、sequence 和 embedding slot 身份映射
  上一轮 pending context，支持继续请求的重排与缩批；使用本轮 block table 和采样参数。
- 可以 prelaunch 时，先用设备侧 accepted tokens 构造并提交本轮 draft，再等待上一轮 D2H、
  更新 host decode 状态和准备本轮 target verify。
- 不能 prelaunch 时，先提交上一轮 host decode 状态，再沿普通 draft 路径执行。
  新请求、JSON 状态、KV swap/copy 等不满足条件的 batch 不会消费预测的 draft。
- 关闭异步调度时，沿用同步收尾；context 写入仍属于当前轮，不依赖下一轮是否存在。

公共 pending-context 提交、异步收尾和空 DP rank 的生命周期由已有 `DFlashWorkerImpl` 承担，
DSpark 和 DFlash2 各自保留具体 draft 计算和逐 batch 准入条件。
异步收尾与 prelaunch 是不同决定：CP/KV split 等 prelaunch 限制不关闭异步收尾。
本轮 DFlash2 的非空 KV transfer batch 暂时保留原同步收尾，避免绕过既有 draft PUSH completion 协议。
prelaunch 的容量检查包含尚未确认的接受长度上界；普通路径完成 host 修正后，仍可独立选择 draft graph。

Python DFlash2 通过 `DFlash2ExecutionPipeline` 编排两个 Runner：

```text
C++ worker：构造本轮 query、准备并跨 TP 同步 Gumbel noise
  → Python ExecutionPipeline
      → decoder Runner：backbone → lm_head → candidate selector
      → proposal Runner：path sampling
  → ModelOutput.draft_proposal → C++ target verify / context 收尾
```

DFlash2 的 query 包含一个 anchor 和 N 个 proposal 位置。`DFlash2DecoderModel` 在同一个 forward 中
执行 backbone、选取 proposal 位置的 hidden states、计算 lm_head 和 candidate selector；
lm_head 仍不计算 anchor logits，selector 仍以该请求的 anchor token 构造首步 predecessor。
这三部分共用一个 Runner 和一张图，不再把 hidden states 拷贝到独立 candidate Runner 的固定输入。
decoder 通过内部阶段输出 `DraftDecoderOutput.candidates` 返回 `[batch, N, K]` candidate ids 和
`[batch, N, K, K]` edge logits，公共输出裁剪按请求数截取候选结果，hidden states 仍按 token 数裁剪。
Pipeline 将候选结果展平为每请求 N 行，交给独立的 proposal Runner；对外的 `ModelOutput` 与 C++ 契约不变。
proposal 模块输出 request 对齐的 `[batch, N]` token ids 和 `[batch, N, vocab]` dense draft probabilities。
两个阶段独立选择 eager/graph；sampling fallback 不影响整个 decoder 的图执行。
`DFlash2DecoderMetadataBuilder` 在 eager 的无采样调用中保留 backbone-only 执行；完整 query block 的图
始终捕获整个 decoder，避免无采样 warmup 缓存的图在后续 proposal 请求中漏掉 lm_head 和 selector。
无采样或 dummy 调用对外仍只返回 backbone hidden states。这里不修改 attention Runner 的输入管理或捕获逻辑。

`DFlash2Qwen3ForCausalLM` 仍持有原有 backbone、lm_head 和 selector，保留 checkpoint 路径和共享权重规则，
通过 provider 接口创建 Pipeline。Executor 优先消费外层模型的 provider，其他模型继续使用 `model.model`。
`DFlash2SamplingMetadataBuilder` 保留原始采样模式与可选温度参数，仅允许无温度修正的全 greedy 请求入图，
并管理 candidate ids 和 noise 的持久 buffer，每步在 capture/replay 外更新。
随机数生成与 TP 同步保持在 C++ prepare 阶段，避免捕获固定噪声。
随机、混合或需要温度修正的请求走 sampling eager，保留动态分支：greedy 不计算 `exp`，
全随机不构造 one-hot，只有混合 batch 使用 `where`，无温度 tensor 时不做温度修正。
penalty、filter 和 JSON grammar 仍在 target verification 应用，不重复应用到 selector 分布。

多 DP 不引入上游执行计划或新的 RPC 字段，执行方式仍由 Python Runner 决定：

- DFlash2 decoder Model 只允许带 `is_dflash_proposal` 标记的展开输入入图。backend 对 draft 的非展开 prefill
  输入正常返回不支持，启动期 warmup 因而走 eager，不会把 16 个请求、128 个 token 的输入误送进普通 decode
  行校验。target 的严格布局校验和原有 backend/Model 短路顺序不变。
- decoder 沿用已有 DP token counts 选择共享 bucket，并保留原有 DP 准入协调。
- sampling Runner 单独汇总本地准入结果；任一有效 shard 不满足 greedy/温度等条件时，所有有效 shard 的
  sampling 都走 eager，decoder 的选择不受影响。没有采样任务的 shard 通过 `skip_execution` 投赞成票，
  但不检查无效的采样输入，也不调用 eager forward 或 graph replay。
- 协调复用 `dp_all_true` 的 host store rendezvous，不新增 NPU collective 或设备标量回读；它仍有主机等待开销，
  不能据此宣称对 prelaunch 重叠或吞吐没有影响。其他 Pipeline 默认不启用这项 sampling 协调。

C++ worker 不再持有 DFlash2 draft ACL Graph；Python 路径直接消费 proposal，非 Python 路径保留原生采样。
NPU Python DFlash2 不再被启动期 `requires_eager_execution` 强制关闭 graph，普通 DFlash 与 native 路径保持原门控。
backbone 复用 `DecodeAclGraphRunner` 的输入管理与捕获逻辑；XFIA 通过 proposal 标记保持展开后逐 token 的 sliding-window
左边界，不把图中的逐 token block table 误当成互相独立的单 token 请求。
上一轮 context KV 写入与本轮设备输入准备均不在 draft 图内。
捕获范围是独立优化，不决定 context 的提交时机，也不要求将 context 后移到下一轮。
