from pathlib import Path
import json,datetime
b=Path(__file__).resolve().parents[1];e=b/'证据';root=b.parent
status=json.loads((e/'matrix_status.json').read_text()) if (e/'matrix_status.json').exists() else {'phase':'PREPARING'}
rows=[]
for run,label in [('stage1_ref','A：全部 reference'),('stage1_swiglu_d_v2','B：仅 D 共享 SwiGLU'),('stage2_post','C：再接 mHC post'),('stage2_post_pre','D：再接 mHC pre'),('stage2_all','E：再接 metadata')]:
 p=e/f'{run}_validated_summary.json'
 if p.exists():
  d=json.loads(p.read_text())
  for r in d['results']:
   m=r['metrics'];rows.append(f'| {label} | {r["input_tokens"]}/256 | {m["ttft_s"]["mean"]*1000:.2f} | {m["tpot_ms"]["mean"]:.2f} | {m["latency_s"]["mean"]:.4f} | {r["output_tokens_over_test_duration"]:.4f} | {r["request_qps"]:.6f} | 5/5，通过原始记录核验 |')
 elif (e/f'{label[0]}_256_verified.json').exists():
  d=json.loads((e/f'{label[0]}_256_verified.json').read_text())
  rows.append(f'| {label} | 256/256 | {d["ttft_ms"]:.2f} | {d["tpot_ms"]:.2f} | {d["latency_s"]:.4f} | {d["output_tokens_s"]:.4f} | {d["qps"]:.6f} | 5/5，单组服务日志已核验 |')
  rows.append(f'| {label} | 2048/256 | — | — | — | — | — | 等待全组结束及验收 |')
 elif run=='stage1_ref' and (e/'partial_256_verification_receipt.json').exists():
  d=json.loads(json.loads((e/'partial_256_verification_receipt.json').read_text())['stdout'])
  rows.append(f'| {label} | 256/256 | {d["ttft_ms"]:.2f} | {d["tpot_ms"]:.2f} | {d["latency_s"]:.4f} | {d["output_tokens_s"]:.4f} | {d["qps"]:.6f} | 5/5，单组原始记录已核验 |')
  rows.append(f'| {label} | 2048/256 | — | — | — | — | — | 等待全组结束及验收 |')
 else:rows.append(f'| {label} | 256/256、2048/256 | — | — | — | — | — | 尚无完整验收结果 |')
sel=e/'shared_swiglu_selection.json';selection='尚未完成 A/B，mHC 正式实验最终采用哪个 SwiGLU 配置待两组结果确定。'
if sel.exists():
 d=json.loads(sel.read_text());selection=f'根据 A/B 两组正式结果，mHC 实验的 D 侧共享 SwiGLU 选择 `{d["D_shared_swiglu"]}`。具体判定与波动阈值见 `阶段一二/证据/shared_swiglu_selection.json`。'
sj_table=''
sj_path=e/'sj9_operator_preflight/mhc.json'
if sj_path.exists():
 sj=json.loads(sj_path.read_text());sj_table='\n另在实际 sj9 GPU 上重复验证：79 项数值检查全部通过。以下计时独立列出，未与 dev6 样本混合。SDK 回显为 MTT S5000、capability 3.1；与系统 gmi 的 X10000 名称一起保存。\n\n| sj9 算子 | T | reference μs | 融合 μs | 局部耗时下降 |\n|---|---:|---:|---:|---:|\n'
 for r in sj['timings']:
  sj_table+=f"| {r['op']} | {r['tokens']} | {r['reference']['median_wall_us']:.3f} | {r['fused']['median_wall_us']:.3f} | {r['wall_reduction_pct']:.2f}% |\n"
 sj_table+='\n原始结果：`阶段一二/证据/sj9_operator_preflight/mhc.json`。\n'
now=datetime.datetime.now().astimezone().isoformat(timespec='seconds')
text=f'''# 2026-11-10：阶段一、二工作总结

更新时间：{now}。当前执行状态：`{status['phase']}`。

**状态说明：** 下表只填入已完成 1 次暖机、5 次正式请求并经原始服务日志核验的数据；空白不代表零耗时，也不代表已完成。编译、原语验证、小模型验证与完整模型性能分别记录。

## 1. 本轮目标及路线

按照今天确定的计划，先把已有共享 SwiGLU 在 sj9/sj10 的原版 1017 模型上完成 A/B，再按 post → pre-combine → metadata/Sinkhorn 接入 mHC，每个算子都做数值与性能测试。

继续基于 1103 的 `xiaopangDeepseeekV41Flash`，本轮不重写原生 C++ 整模型。Python 组织模型，C++ 绑定调用已编译 MUSA kernel；这次复用的三个 mHC 原语来自 xLLM 内已有 V4 实现，不能一概称为 MATE 实现。

后续 Python、专家 FP4/block32、CSA2、Engram/CED、Graph，以及 C++ 局部/整模型对照的详细顺序，已写入[今晚待运行的 goal](今晚待运行的goal.md)。该文件没有自动创建新 goal。

## 2. 场地、资产与版本

| 项目 | 本次实际值 |
|---|---|
| 部署 | sj9 为 P、sj10 为 D；各 8 GPU，1P1D，eager |
| 硬件回显 | sj9/sj10 的 gmi 显示 X10000、每卡 81920 MiB、driver 3.3.8-server；保留原始回显，不用 dev 机驱动信息替代 |
| 真实子模块并行 | Attention TP8；路由专家 EP8、专家内 TP1；共享专家当前完整复制 |
| 模型 | `/data/nfs_shared/models/DeepSeek-V4.1-Flash-1017`，两机同一 NFS 可见 |
| 模型版本 | HF `deepseek-ai/DeepSeek-V4.1-Flash-1017`，commit `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| 权重形式 | 官方原版 FP4 专家权重；未换成 W4AFP8-mt 的 INT4 group128 版本 |
| 仓库 | https://github.com/fay85/xllm-musa/tree/xiaopangDeepseeekV41Flash |
| 共同父基线 | `0cfd82700e164e8aa7951064b652925dd6b24d18`（1103） |
| 阶段一起点 | `34815f6ac23de6c3145a3392b7649874a37d4f27` |
| 镜像 | `registry.mthreads.com/presale/devtech/xllm:1102-dev` |
| 镜像 ID | `sha256:09b8f5f4a467fc73332dee57b36a2e5610ba3264357dfc87f103cf7bd7430be3` |
| 依赖 | Torch/TorchMUSA 2.9.1，MATE 0.2.7；其余依赖快照见 manifest |
| 控制服务 | 镜像预置 `/opt/xllm_service/bin/xllm_master_serving` 与 etcd；本轮未重编控制服务。文件 SHA 另见 `阶段一二/证据/service_preinstalled_components.json` |
| 阶段一 xllm SHA256 | `8d5e8c5e83227021672a472f749782b4426654abf264f10f815923e57074caa6` |
| 新 mHC xllm SHA256 | `862d3ab3a8a1eab99326c353782af0b7a170041e53174063735a27051ae0a5af` |

两个旧 GLM 容器 `xiaopang1105_glm_prefill_sj9`、`xiaopang1105_glm_decode_sj10` 已按用户明确许可归档并正常 TERM 停止，最终均 Exited 143、OOMKilled=false，八卡显存释放后才启动本任务。没有使用 SIGKILL，没有修改驱动或重启服务器。

本轮在 dev6 的我方构建容器内重新编译 xllm 及 MUSA 数值测试扩展，具体命令保存在 `阶段一二/脚本/build_stage2.sh`。构建复用镜像内的 Rust tokenizer/safetensors 产物和 vcpkg 依赖，不是把全部第三方依赖从源码重建。运行时三处生产代码与冻结源码 SHA 一致；后来扩展的 79 项验证脚本另见 `stage2_test_update_manifest.json`。

本轮使用 `xiaopang1110_v41_prefill_sj9`、`xiaopang1110_v41_decode_sj10`；临时单卡回归使用 dev6 的空闲 GPU0，独立容器退出后归档清理。

## 3. 已取得的正确性证据

- 1017 全模型通过两道基础题：`1.3+2.9 → 4.2`、`7×8 → 56`，正常 stop，chat 模式，temperature=0，无思考输出。第一题生成 4 token，第二题 2 token；它们只构成基础功能检查，不代表公开评测集对分完成。
- 原版 tokenizer/Engram sidecar 核验通过：129280 个 token ID、99092 个压缩 ID；token_map 差异 0，multipliers 一致。
- mHC 原语测试 39 项通过；再加真实 1017 的层 0/1/14/39、attn/ffn 参数的 metadata 及 pre/post 测试 40 项，合计 **79 项通过**。真实层参数输入仍是随机 hidden，不冒充整模型实际 hidden 的逐层验证。
- 现有 CPU mHC 回归：26 项通过，6 项因 NPU 融合路径不可用而跳过；C++ 格式检查使用 clang-format 20.1.6，通过。
- 六层随机小模型，2 条序列各 9 token，含 CSA2/Engram：reference、post、post+pre、all 均通过整段 prefill 与逐步 decode 对照；相邻配置 hidden/logits 逐元素一致，greedy token 一致率 100%。
- 全模型固定 token 前缀及跨 P/D 续算的数值检查**尚未完成**。Completion 探针遇到 V2 correlation 错误；Chat 控制试验发送 21 个原始 ID，却按 messages 编码成 17 token 并回答 messages 中的问题，不能作为相同前缀比较。两种失败都不是模型数值结论，详见协议缺口文档。

具体每项覆盖、数学公式、文件改动、验收误差与限制见 [mHC 接入与数值问题说明](阶段一二/mHC接入与数值问题说明.md)。

## 4. 每个新算子的性能结果

下面是 **dev6 单卡微基准**，H5120、hc_mult=4；先暖机 10 次，再测 5 组，每组 30 次调用，采用每次 wall time 中位数。包括 binding 和输出分配；GPU event 原始数据也保留。这不是整模型 TTFT/TPOT。

| 算子 | T=1，reference→融合 μs | T=256，reference→融合 μs | T=2048，reference→融合 μs |
|---|---:|---:|---:|
| mHC post | 99.534 → 13.955 | 384.988 → 24.845 | 3055.413 → 218.663 |
| mHC pre-combine | 65.960 → 11.565 | 114.308 → 28.330 | 1000.765 → 227.305 |
| mHC metadata/Sinkhorn | 1584.088 → 47.179 | 1603.986 → 46.883 | 1612.904 → 91.497 |

{sj_table}
metadata 局部提升的实现原因是把许多小的归约/除法及 Sinkhorn 迭代合到已有 GPU kernel 内。没有新抓 trace，不能把该判断写成已由 trace 直接观察的结果。完整模型仍有逐专家解量化/计算等主耗时，局部下降 97% 不能写成 TPOT 下降 97%。

## 5. 1017、1P1D 的完整性能表

协议：每侧 8 卡，C=1，eager，temperature=0，chat 模式，关闭跨请求 prefix cache、关闭推测解码；prefill chunk=256。固定 prompt 要求用英语解释记录实验的价值，并用无指令填充补足长度；两组分别经官方 tokenizer 和服务原生计数确认 256/2048 输入。输出强制 256 token，属于固定长度性能负载，不是对这篇回答的精度评分。每组暖机 1 次不计入统计，正式 5 次；用未修改的 EvalScope 1.14.1 perf，逐条核对 P/D 原生日志、实际 input/output token 与成功终止。

配置对应关系如下。A/B 已完成，B 的两组收益达到事先设定的实用波动阈值，因此 C/D/E 保留 D 侧共享 SwiGLU 融合；P 侧仍为 reference。该开关只在本轮配置开启，源码默认值仍为 reference。

| 配置 | P 共享 SwiGLU | D 共享 SwiGLU | 两侧 mHC 融合项 | 二进制 |
|---|---|---|---|---|
| A | reference | reference | 无 | 阶段一 |
| B | reference | v4_fused | 无 | 阶段一，与 A 相同 |
| C | reference | v4_fused | post | 阶段二 |
| D | reference | 同 C | post、pre-combine | 阶段二，与 C 相同 |
| E | reference | 同 C | post、pre-combine、metadata | 阶段二，与 C/D 相同 |

| 配置 | 实际输入/输出 | 平均 TTFT ms | 平均 TPOT ms/token | 平均端到端 s | 输出吞吐 token/s | QPS | 正式请求验收 |
|---|---|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

{selection}

TPOT 采用该 EvalScope 版本的 `(请求端到端时延−TTFT)/(输出 token−1)`，包括请求尾部开销。输出吞吐和 QPS 分别用输出 token 总数、成功请求数除以 EvalScope 记录的组测试持续时间；不能直接用 `1/TPOT` 代替输出吞吐。原始 CSV/JSON 同时保留中位数、范围和样本标准差。

配置审计见 `阶段一二/证据/configuration_matrix_audit.json`。A 的 `/work/repos/xllm-musa` 是 B 使用的 `/work/01_sources/34815f6a` 的软链接，已核对实际指向。并行、缓存、Graph、采样设置保持一致。B→C 同时使用了新增 mHC binding 的二进制及后端选择代码，属于一次集成对照；C→D→E 则共用同一个阶段二二进制。

工具界面可能根据流式 chunk 数推断“Decoded Tok/Iter / Spec. Accept Rate”；当前显式 `num_speculative_tokens=0`，不能把这种显示解释成已经开启 MTP 或得到了真实接受率。

## 6. 遇到的问题与处理

1. **旧服务占卡：** 等用户确认用途后，归档指定 GLM 容器并正常停止，再确认八卡空闲。没有操作他人容器。
2. **第一次 EvalScope 未启动成功：** 隔离客户端环境缺少 uvicorn，尚未发请求就报错。只给我方复制出的测速 venv 补齐 perf 依赖；模型服务镜像环境未被改动。失败保存在 `03_benchmarks/stage1_ref`，有效重跑在 `stage1_ref_v2`。
3. **mHC 极端输入差异：** 初次与原 MUSA PyTorch 参考比较失败。保存同一输入后发现，原生 kernel 对 FP64 最大差约 1.86e-7，旧参考约 0.0046243；差异定位到该环境中一处极小列和的归约。改用独立 FP64 oracle 验收饱和输入，保留旧参考差异，未放宽阈值。CPU oracle 仅用于测试。
4. **真实参数中转：** 只从 1017 提取 24 个 mHC 张量作层测试，15.7 MB，两跳 SHA 一致。没有搬运或修改完整模型。一次本机 SFTP 预取停滞改为有限分批读取，未影响远端模型服务。
5. **启动注册时序：** B 的模型 HTTP 端口刚就绪时，基础题收到 HTTP 503；尚无模型答案，原始失败已归档。给客户端添加最多 120 秒的 503 可用性等待；其他错误和答错仍立即失败，不自动反复挑答案。
6. **耗时边界：** 当前共享/路由专家的高性能路径尚未完整接入；完整模型单次 256 输出仍需要数分钟。微基准、暖机与正式 5 次数据严格分开，不提前填“已完成”。

7. **配置切换端口检查：** A 的 16 个 rank 正常退出后，立即 bind 检查曾被未完全释放的端口拦住；无残留服务监听。补上有界等待并修正远端失败必须使本地 launcher 失败，再启动 B。未停止任何额外进程。
8. **固定前缀探针未完成：** `/v1/completions` 的非流式原始 token 请求返回 HTTP 400：`Stale or mismatched V2 output identity`。保留响应与请求；此为服务协议验证缺口，不能写成模型数值不一致或数值验收通过。基础题和性能均使用流式 chat，已经单独验收。
9. **容量拒绝：** B 首轮两道基础题通过，随后固定前缀探针触发 V2 output 缺少 attempt_seq 的协议错误。之后测速全部返回容量不足的 HTTP 503，没有有效生成，已判 FAIL；两者因果关系仍需验证。正常结束此轮 16 个 rank，原始记录保留，B v2 去掉该已知不兼容探针后重试，两组暖机和正式 5 次均通过验收。官方工具可能在请求全失败时仍返回 CLI 0，因此补上每阶段数据库检查，暖机失败就停止后续请求。
10. **日志存档：** 初次验证对持续追加的 rank0 日志取 hash，之后探针/正常退出又追加了日志。已按原 hash 找回完全一致的字节前缀，固化为只读证据快照；原始 hash、数值均未改动。后续验证直接先保存快照再取 hash。
11. **本机 SSH 短暂中断：** 19:33 本地调度查询遇到 authentication timeout 而退出，19:36 已恢复连接。远端 C 测速独立运行，期间完成 256/256 的五次请求并进入长输入暖机，没有重启 VPN 或模型。调度从现有 C 恢复，仅为只读查询增加有限重试，不自动重试启动/停止操作。客户端实际运行在 sj10，这次本机 SSH 中断不能作为性能变化的原因。

12. **客户端完成标记读取竞争：** D 的两组数据已经完成，但调度第一次查询完成标记时尚不存在，第二次查询已看到客户端正常退出码，误报提前退出。修正为看到退出码后再次读取完成标记，仍不以 CLI 0 单独验收；复核两组原始数据库后直接收尾，没有重跑已完成请求。
13. **显存释放时序：** D 的 16 个 rank 正常退出后，E 的 sj9 零占用检查一度未通过，sj10 则已启动。后来确认 sj9 八卡均为 0 MiB/0% 且无 GPU 进程，只启动缺失的 sj9 角色，再启动客户端。启动器增加最多 60 秒的只读空闲等待；不停止任何现有进程，已启动的 sj10 没有重复启动。

## 7. 没有在本轮完成、留给后续的事情

完整 Mega-mHC 融合、原始 FP4 + block32 专家 GEMM、完整 GPU 路由/combine、Indexer 官方量化与 CSA2 全部高性能算子、CED bounded replay、Engram 优化、Graph/DSpark/MTP、原生 C++ 整模型，以及 H20/H200/60% SOL 对照，都不能因为本轮三个 mHC 原语通过而算作完成。

[今晚待运行的 goal](今晚待运行的goal.md) 已将 Python 主线和 C++ 路线分别展开，列明来源、依赖、验证与性能比较前提。后续继续参考家明 V4、开源 SGLang 和京东交付实现，优先解决专家计算主耗时；原版 FP4 与交付 INT4 方案分开验收。

## 8. 原始记录在哪里

**sj9/sj10 共用 NFS：**

```text
/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops/
```

`00_manifest` 是版本/旧容器归档；`01_sources` 是冻结源码；`runs` 是各 rank 与控制服务日志；`02_correctness` 是问答/前缀探针；`03_benchmarks` 是 EvalScope SQLite、逐请求记录与统计；`04_reports` 保存报告。

**dev6 单卡回归和构建：**

```text
/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261105跑通PD分离/20261110_ops/stage2/
```

**本地：** 当前目录的 `阶段一二/证据`、`阶段一二/脚本`。正式结果汇总为 `<run>_validated_summary.json`；mHC 原始检查与计时在 `mhc_v2.json`、`dev6_mhc/mhc_v4.json`；失败及后续诊断也保留。编译产物/源码完整性通过后，已清理 `阶段一二/传输缓存` 中 278,748,094 bytes 中转文件，保留 manifest 与清理记录。

最终提交链接与给家明的第一人称说明将在所有本轮验收完成后补入；本轮不提 MR/PR。
'''
(root/'今日工作总结.md').write_text(text)
print(root/'今日工作总结.md')
