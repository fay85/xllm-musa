from pathlib import Path
import json,datetime,shutil
b=Path('王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/20261116后续如何正确推进并知道边界大概在哪里/追平sglang甚至超越它');e=b/'02_实验记录/实验_042';v=json.loads((e/'03_逐请求与trace分析.json').read_text());a=v['off'];z=v['on'];m=lambda x:x['summary'];rows='\n'.join(f"|{label}|{m(x)['mean_TTFT_ms']:.6f}|{m(x)['mean_TPOT_ms_per_token']:.6f}|{m(x)['mean_E2E_s']:.6f}|{m(x)['output_tokens_per_latency_sum_second']:.6f}|" for label,x in [('同ELF关闭FP8投影',a),('同ELF仅D开启FP8投影',z)])
text=f"""# E42：D端原生FP8投影已接通，节省显存但本轮不晋级速度

结论 **READY_DEPENDENCY，不替换E39累计最佳**。两组正式测速、独立trace、ON长短请求切换及正常退出已完成。仅通过基础功能检查，未完成模型精度验收。21.57ms/token、39.45token/s的总目标尚未达到。

## 1. 为什么做、做到了哪一层

E40的真实文件头预算表明，若把Attention和共享专家完整复制成BF16，后续owner/DP执行的单卡容量不足。因此本项接通原始FP8权重持有、scale和原生投影，作为布局变化的前置；没有把微基准或显存节省直接算成整模型提速。

|层次|本项改动|保持的边界|
|---|---|---|
|权重加载|checkpoint新增原始FP8与UE8M0 scale的CPU同边界TP分片；scale解为FP32后H2D|不修改权重文件、不保留这些投影的BF16载体；检查block32对齐|
|模型Attention|target/draft的QA、QB、KV、OB传入FP8权重与scale|O-A、Indexer、压缩、Engram、LM-head保持原路径|
|共享专家|gate/up/down传入FP8投影，SwiGLU及归约位置不变|路由专家仍W4AFP8、EP8；没有重复归约或删除共享专家|
|后端|10种形状×1至32行的320项AOT；C++选择当前MUSA stream；D侧超过32行时分块拼回|所有投影计算在GPU；Python用于离线编译/客户端，不是在线Python模型forward|
|显式选路|XLLM_DSV41_DENSE_PREQUANT_DIR默认关闭，候选只在D开启|P、TP8/EP8、Graph和proposal3不变；不是已完成真正DP/LL接入|

涉及文件可按本次commit diff核对：deepseek_v41_checkpoint.h、deepseek_v41_native.h、deepseek_v41_dspark_native.h，以及layers/musa/deepseek_v41下feed_forward.h、attention_projection.h、attention.h、moe.h；底层dense_block32.cpp/.h和离线build_dense_prequant_aot.py。构建/测试的完整源码指纹在01_代码与构建与前置证据包内。

## 2. 小检查与失败处理

198项真实Dense投影、44项改变输入的Graph、24项共享MLP、12项共享Graph通过；CPU分片覆盖rank0/3/7、行/列切分与权重scale对应。详见01_前置验收与产物.md。

测试模块首版链接缺SwiGLU符号，改为链接实际编译对象；第一版GPU脚本在投影检查结束后发现旧模块没有共享MLP绑定，保留失败，仅补做剩余共享检查。没有放宽阈值，也没有把测试脚本错误写成模型或驱动故障。OFF客户端继承的文字source_manifest曾指向旧E37，已保留原文件并新增identity_correction.json；实际ELF SHA正确，未修改请求或成绩。

## 3. 正式性能：容量改善与速度结果分开

SJ9 P/SJ10 D，各8卡，1P1D、Attention TP8/专家EP8、target q4完整Graph、proposal3、C1，同W4AFP8-mt1017。每组EvalScope1.14.1独立1暖机＋5正式，profiler OFF；实际256输入/256输出，全部success、finish_reason=length，客户端DB与服务端request_id/sequence_terminal交叉核验。两组默认的packed QKV/shared gate-up本来就关闭，未额外变更。

|配置|TTFT ms|TPOT ms/token|E2E s|sum(tokens)/sum(latency) token/s|
|---|---:|---:|---:|---:|
{rows}

端到端输出下降{-v['difference']['output_increase_pct']:.4f}%，TPOT增加{-v['difference']['TPOT_reduction_pct']:.4f}%。差距超过3%且五次波动小，不为负收益重复完整反向一轮。E39仍是累计最佳；OFF35.06token/s是原配置重测，不能记作新优化收益。

D端加载阶段可用显存由约10.83GiB升到13.60GiB；短请求Graph捕获后GMI每卡从74362MiB降到71053MiB，减少3309MiB（3.23145GiB）。它包括分片持有和allocator实际变化，不等于E40在完整DP假设下估算的5.12395GiB。后续还须单独预算owner完整权重、LL buffer、缓存、Graph与加载峰值，不能据此直接宣称新布局一定装得下。

## 4. 为什么不能只看GEMM速度

控制组每题accepted/proposed=130/375、125轮；候选126/390、130轮。实际交付仍256token，最后一轮按max_tokens截断。首token后墙钟时间/轮约{a['scope_notes']['decode_wall_ms_per_proposal_round']:.4f}ms→{z['scope_notes']['decode_wall_ms_per_proposal_round']:.4f}ms；这是draft、verify、缓存和主机合计，不是单算子时间。FP8载体与BF16载体的舍入路径不同，输出及接受量变化已记录；按本轮要求不展开模型精度原因分析。

真实rank0末次q4 Graph节点从{a['D_rank0_last_graph']['kernels']}变成{z['D_rank0_last_graph']['kernels']}，GPU span从{a['D_rank0_last_graph']['span_us']/1000:.4f}ms变成{z['D_rank0_last_graph']['span_us']/1000:.4f}ms。它们是独立采集下的路径/工作量线索，不是无profiler的精确收益归因。分开查看CPU C++/ATen入口、Graph内部GPU节点、投机有效输出；不能把入口COUNT火焰图宽度当CPU耗时，不能把kernel数量变化直接换算成TPOT收益。

开启组末次Graph明确出现280次 `quantize_block32_once`、280次 `prequantized_block32_gemm` 和280次 `dense_block32_reduce`，对应40层×7个投影，证明新选路实际进入GPU图。两组图外原生C++/ATen入口均为634，没有因为使用FP8就减少主机入口数；变化主要在图内投影节点与投机工作量。独立256/8短请求关闭profiler约0.49～0.69秒，采集时约18秒，扰动明显，不用这些采集时间计算正式速度占比。

![关闭FP8：真实CPU与GPU时间线](04_trace/FP8投影_OFF/Perfetto图/真实CPU区间与GPU时间线.png)

![开启FP8：真实CPU与GPU时间线](04_trace/FP8投影_ON/Perfetto图/真实CPU区间与GPU时间线.png)

![开启FP8：真实C++入口计数火焰图](04_trace/FP8投影_ON/Perfetto图/完整实际C++调用链_ENTRY_COUNT.png)

两组各16rank GPU事件和C++ sidecar按record ID/tid关联，并用本次ELF符号化。addr2line对部分DWARF FORM0x23提示不支持，函数名有效，未宣称所有源码行号完整。

## 5. 长输入、退出与证据

ON实际2048/256单次冒烟{z['long_smoke']['latency_s']:.6f}s，首内容{z['long_smoke']['first_content_s']:.6f}s，含长图桶首次捕获，不能作为正式长输入性能；之后4.2、56、256/8全部通过。两组共32rank正常exit0，没有SIGKILL，没有操作他人任务或修改驱动/服务器。08:06:46两台GPU空闲已核验。

- 共同仓库个人分支：xiaopangV41FlashCppFromSglangtoXllm；不提MR。
- 实际ELF：/expt/artifacts/native_sg438_fp8dense_v1/bin/xllm，SHA256 7389984e8a774025f574b79cc2a50b00fd50df09eb91773636c7bcb5fa50934d。
- NFS根：/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116；runs/p42_fp8dense_off_v1及runs/p42_fp8dense_on_v1。
- 正式DB：各run的formal_client/formal/deepseek-v4.1-flash-1017-w4afp8-mt/parallel_1_number_5/benchmark_data.db。工具窗口吞吐/QPS单列保留。
- 完整16rank trace保留NFS profiling/p42_fp8dense_off_v1或on_v1/sj9|sj10；本地原始证据_off/on包含rank0trace、全rank校验清单、DB、日志和实际依赖SHA。
- 归档回执各自记录总大小和SHA，前置包199232B SHA25d5dc20b53cc9264b77c45d347773b5a8475ddffdf380a0671258790dcb9085。

## 6. 下一步与退出条件

保留FP8选路作为有证据的容量前置，不改变默认最快配置。下一步把E38已通过的DeepEP/MATE专家接到完整模型：先显式C1 owner Attention/shared，其他rank参与空token dispatch与本地专家；Embedding/Engram/LM-head必要TP8 collective仍完整，owner输出与draft辅助hidden必须正确传播，不重复求和。先短小通信/缓存检查，再做同口径正式对照。

若这条组合仍无稳定端到端收益或容量预算不通过，保留证据并换路线；不会因单项已经写完而宣布总目标成功。最终同一artifact/config三轮验收、正式2048/256及同窗口SGLang B复测仍未完成。
"""
(e/'05_结论与下一步.md').write_text(text)
print('report generated')
