"""Write final prose only after the five configurations were validated and stopped."""
from pathlib import Path
import datetime
import json
import subprocess
import sys

base = Path(__file__).resolve().parents[1]
evidence = base / '证据'
parent = base.parent
root = '/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops'
branch = 'https://github.com/fay85/xllm-musa/tree/xiaopangDeepseeekV41Flash'
state = json.loads((evidence / 'matrix_status.json').read_text())
assert state['phase'] == 'ALL_BENCHMARKS_VALIDATED_AND_RANKS_STOPPED', state
names = ('stage1_ref', 'stage1_swiglu_d_v2', 'stage2_post', 'stage2_post_pre', 'stage2_all')
data = {name: json.loads((evidence / f'{name}_validated_summary.json').read_text()) for name in names}
assert all(d['status'] == 'PASS' and len(d['results']) == 2 for d in data.values())
subprocess.run([sys.executable, str(base / '脚本/compare_validated_runs.py')], check=True, stdout=subprocess.DEVNULL)
subprocess.run([sys.executable, str(base / '脚本/render_today.py')], check=True, stdout=subprocess.DEVNULL)

def row(name: str, length: int) -> dict:
    return next(r for r in data[name]['results'] if r['input_tokens'] == length)

def metrics(name: str, length: int) -> str:
    r = row(name, length)
    m = r['metrics']
    return (f"{length}/256：TTFT {m['ttft_s']['mean'] * 1000:.2f} ms，"
            f"TPOT {m['tpot_ms']['mean']:.2f} ms/token，端到端 {m['latency_s']['mean']:.4f} s，"
            f"输出吞吐 {r['output_tokens_over_test_duration']:.4f} token/s，QPS {r['request_qps']:.6f}")

def gain(before: str, after: str, length: int) -> float:
    return 100 * (1 - row(after, length)['metrics']['tpot_ms']['mean'] / row(before, length)['metrics']['tpot_ms']['mean'])

comp = json.loads((evidence / 'operator_full_model_comparisons.json').read_text())
e_comps = [r for r in comp['comparisons'] if r['after'] == 'stage2_all']
assert len(e_comps) == 2
output_note = ('E 与 D 两组各五次文本均一致。' if all(r['all_repeated_output_text_identical_and_equal'] for r in e_comps)
               else 'E 与前置 D 的输出文本未在两组都保持完全一致，性能变化还可能伴随专家路由变化。')
e_perf = (f"最后加入 metadata 后，相对 post+pre 配置，256/256、2048/256 的 TPOT 分别下降 "
          f"{gain('stage2_post_pre', 'stage2_all', 256):.2f}%、{gain('stage2_post_pre', 'stage2_all', 2048):.2f}%"
          '（负值表示变慢）。')
raw_short = root + '/03_benchmarks/stage2_all/input256_output256/measured/deepseek-v4.1-flash-1017/'
raw_b = root + '/03_benchmarks/stage1_swiglu_d_v2/input256_output256/measured/deepseek-v4.1-flash-1017/'
report = f'''# 给家明的本轮汇报

家明，我继续基于 1103 的 [xiaopangDeepseeekV41Flash]({branch}) 推进，完成了原版 1017 在 sj9/sj10 的双机 1P1D 验证，以及共享 SwiGLU 和三项 mHC 原语的逐项接入、对照测速。P/D 各 8 卡，eager、并发 1；实际是 Attention TP8、路由专家 EP8/专家内 TP1，共享专家完整复制。源码提交在这个共同仓库的个人分支上，不提 MR。上一项 SwiGLU 提交是 [34815f6a](https://github.com/fay85/xllm-musa/commit/34815f6ac23de6c3145a3392b7649874a37d4f27)，本轮 mHC 代码和全部实验记录纳入本次提交，最终提交网址另存于本地及 NFS 的“最终提交与交付记录.md”。

这次不是另写一套 GPU 算子，而是复用你 V4 在 xLLM 里已有的 MUSA kernel：共享专家的 clamped SwiGLU，以及 mHC 的 post、pre-combine、metadata/Sinkhorn。我增加了 C++ 绑定、Python 后端选择与 V4.1 调用，保留 V4.1 跨子层传递 pre_mix 的顺序。它们在 GPU 上执行；模型主体仍是 Python forward。这解决了 V4 已有原语尚未接到 V4.1 上的问题，不能说 V4.1 的完整 Mega-mHC、FP4 专家路径也一起完成了。

每项接入均测了两组输入输出，每组 1 次暖机、5 次正式请求，用未修改的 EvalScope 1.14.1，并核对两端服务的真实 token 数。只接 D 侧 SwiGLU 时，TPOT 分别改善 0.73% 和 0.61%，回答文本与 reference 一致；post/pre 的单算子计时变快，但整模型没有获得增量收益。{e_perf} 三项 mHC 全开后的结果是：{metrics('stage2_all', 256)}；{metrics('stage2_all', 2048)}。{output_note}

数值方面完成了实际 sj9 上 79 项原语/真实层参数检查，六层随机小模型的 prefill/decode 和四种配置对照也通过，两道基础题每个配置均通过。但全模型开放式回答在部分配置间发生变化，固定 token 前缀的进一步数值检查又遇到了服务接口问题，所以现在只能说接入与功能/性能实验完成，不能说完整精度已经验收。融合开关保留为显式选择，不改变源码的 reference 默认值，也没有声称已由 trace 证明加速原因。

下一步先修复固定前缀验证入口、查清这些数值差异；同时优先确认原版 FP4 group32 权重加 block32 激活的专家 GEMM 接口，再接 GPU 路由、专家分组/combine。随后做 CSA2 的 norm/RoPE/Indexer/稀疏 attention 和缓存复用，再补 Engram、CED 尾部 replay 与完整 mHC 融合，最后推进 Graph/MTP。C++ 路线先做同一 GPU 算子的局部调用对照，待两条路线算子与模型语义对齐后再比整模型；不把交付 INT4 group128 权重与原版 FP4 混在一张性能表里。

**256/256 原始报告（当前 E 配置）：**

```text
{raw_short}
```

目录内的 `performance_summary.txt`、`perf_report.html` 是原工具报告；`parallel_1_number_5/benchmark_data.db` 保存逐请求数据。共享 SwiGLU 的 B 配置原始报告在：

```text
{raw_b}
```

五种配置的汇总、源码/镜像/二进制校验值、失败尝试和未完成项，见同目录 `今日工作总结.md` 与 `阶段一二/逐项算子完整模型对照.md`。原始包也随本提交放入 `docs/experiments/20261110_deepseek_v41_mhc_1p1d/raw_runs/`。
'''
(parent / '给家明的阶段性汇报.md').write_text(report)
summary_path = parent / '今日工作总结.md'
summary = summary_path.read_text()
summary = summary.replace('**状态说明：** 下表只填入已完成 1 次暖机、5 次正式请求并经原始服务日志核验的数据；空白不代表零耗时，也不代表已完成。编译、原语验证、小模型验证与完整模型性能分别记录。',
    '**完成边界：** 五种配置的两组性能实验全部完成，共 50 次正式请求、10 次性能暖机，均核对 P/D 原始 token 计数。每个配置两道基础题通过。阶段一已完成；阶段二的三个原语接入、分层测试与逐项性能实验已完成，但真实整模型固定前缀数值等价性尚未验收，不能称为完整精度通过。所有实验 rank 及控制进程已正常退出，最终显存释放情况另见终态记录。')
conclusion = f'''## 5.1 如何使用这些结果

共享 SwiGLU 的 A/B 两组 TPOT 分别下降 0.73%/0.61%，超过预设的实用波动阈值，且固定请求的输出文本一致，因此在后续 C/D/E 实验中保留 D 侧融合。这里只证明本轮顺序样本的结果，不把五次测试当成所有负载的一般保证。

post 和 pre-combine 的局部算子明显更快，整模型 TPOT 却没有正向增量，因此不将它们列成已确认的端到端优化。{e_perf} 相对最初 A，E 的 TPOT 分别下降 {gain('stage1_ref', 'stage2_all', 256):.2f}%、{gain('stage1_ref', 'stage2_all', 2048):.2f}%。{output_note}

C/D 等配置的开放式输出已经变化；缺少真实模型同前缀 logits 对照时，无法分离浮点舍入、潜在接入误差与路由负载变化。当前三项 mHC 均作为可选实验实现提交，源码默认仍为 reference。不能因为它们通过基础题、原语容差和性能请求验收，就默认认为完整模型精度已经等价。

本次没有新抓 trace。metadata 将多次小归约和 Sinkhorn 迭代放进既有 kernel，是源码和微基准能支持的解释；实际整模型收益只采用表中实测值，尚不宣称 kernel launch/CPU 空隙已由 trace 定量定位。

'''
summary = summary.replace('## 6. 遇到的问题与处理', conclusion + '## 6. 遇到的问题与处理')
summary = summary.replace('最终提交链接与给家明的第一人称说明将在所有本轮验收完成后补入；本轮不提 MR/PR。',
    '本轮源码、测试和原始实验压缩包提交到共同仓库个人分支，未创建 MR/PR。给家明的说明见 `给家明的阶段性汇报.md`；提交完成后的实际 SHA/网址另写入本地与 NFS 的 `最终提交与交付记录.md`，避免在 commit 内容里伪造其自身 SHA。')
summary += f'\n256/256 当前 E 原始报告目录：`{raw_short}`。\n'
summary_path.write_text(summary)
plan_path = parent / '今晚待运行的goal.md'
plan = plan_path.read_text()
marker = '## 0. 本轮结束后的起跑点\n'
assert marker not in plan
insert = f'''{marker}
A/B/C/D/E 的 256/256 与 2048/256 两组均已完成 1+5，原语接入和对应性能证据都已保存，后续不重复这些未改动的实验。共享 SwiGLU 在 D 侧有约 0.6%–0.7% 的本轮 TPOT 收益；post/pre 未获得整模型增量收益。{e_perf}

下一轮最先处理真实整模型的同前缀数值缺口：修复或绕开当前错误的原始 token 输入验证入口，保留本轮已经产生分歧的固定 prompt 和前缀，定位首次 hidden/logits 差异。C/D 等输出变化没有充分归因，不应直接全局打开 mHC。源码默认 reference，已编译融合实现以开关选择。之后再按第 5 节推进专家高性能路径，避免把新量化差异和本轮未定位的差异叠在一起。

'''
plan = plan.replace('## 1. 要完成什么', insert + '## 1. 要完成什么')
plan_path.write_text(plan)
message = f'''接入 V4.1 三项 mHC GPU 原语，完成 1017 双机逐项测速并保留数值缺口

家明，我继续在基于 1103 的 xiaopangDeepseeekV41Flash 上推进，
复用 V4 已有 MUSA kernel，补上 mHC post、pre-combine、metadata/Sinkhorn
的 C++ 绑定与 V4.1 Python 调用，保留 shifted pre_mix 顺序。
模型主体仍是 Python forward；这次不是完整 Mega-mHC 或原生 C++ 模型。

我在 sj9(P)/sj10(D) 使用原版 1017、两端各八卡，eager、并发 1，
Attention TP8、路由专家 EP8/专家内 TP1、共享专家复制。
每项接入均测 256/256 和 2048/256，各暖机 1 次、正式 5 次；
五种累计配置共 50 次正式请求，用未修改的 EvalScope 1.14.1 验收。

只接 D 共享 SwiGLU，TPOT 分别改善 0.73%/0.61%，输出文本一致。
post/pre 单算子更快，但整模型没有增量收益；负结果也保留。
{e_perf}
三项 mHC 全开：
- {metrics('stage2_all', 256)}。
- {metrics('stage2_all', 2048)}。

我完成了 sj9 上 79 项原语/真实层参数检查、六层小模型对照、
各配置两道基础题；但部分全模型回答文本发生变化。
Completion 的 V2 correlation 和 Chat 的原始 token 输入控制未通过，
所以真实整模型同前缀数值检查尚未完成，不能称为完整精度验收。
融合保持显式开关，reference 默认不变；本轮没有抓新 trace。

下一步先补同前缀验证和差异定位，再确认 FP4 group32 + block32 GEMM，
接 GPU 路由/grouped GEMM/combine，随后 CSA2、Engram/CED、完整 mHC，
最后 Graph/MTP；C++ 路线在相同算子和模型语义下单独对照。

代码、完整性能报告、原始 SQLite/HTML/rank 日志压缩包、失败证据及计划：
docs/experiments/20261110_deepseek_v41_mhc_1p1d/
原始 NFS 根目录：{root}/
256/256 E 报告：03_benchmarks/stage2_all/input256_output256/measured/
deepseek-v4.1-flash-1017/

所有实验 rank/控制进程正常退出；未改驱动、未重启、未操作他人容器。
只提交共同仓库个人分支，不创建 MR/PR。
'''
(parent / 'commit_message.txt').write_text(message)
print(json.dumps({'status': 'FINAL_PROSE_GENERATED', 'at': datetime.datetime.now().astimezone().isoformat(), 'report': str(parent / '给家明的阶段性汇报.md')}, ensure_ascii=False))
