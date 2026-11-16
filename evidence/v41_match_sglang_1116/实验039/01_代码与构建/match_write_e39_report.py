from pathlib import Path
import json,datetime,shutil,hashlib
w=Path.cwd();g=w/'王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做';o=g/'20261116后续如何正确推进并知道边界大概在哪里/追平sglang甚至超越它';e=o/'02_实验记录/实验_039';d=json.loads((e/'03_逐请求与trace分析.json').read_text());off=d['off']['summary'];on=d['on']['summary']
report='''# E39：D端静态专家激活量化，使当前请求输出速度提高7.21%

本轮以同一原生C++ ELF，比较动态group128激活与SGLang采用的checkpoint静态scale。两边各1次正式暖机后5次正式256/256，全部成功；ON进入新的阶段累计最佳。**未达到最终21.57ms/token、39.45token/s门槛，最终三轮验收尚未完成。仅通过基础功能检查，未完成模型精度验收。**

## 1. 为什么做、到底改了什么

SGLang冻结源码 `7c97bb0e6d8b0f8fd41cd797a77b54f7162c79f6` 的 `srt/layers/quantization/w4afp8.py:98-109` 将普通Linear设为dynamic、MoE设为static；专家权重仍为INT4/group128。本地原生模型已有默认关闭的静态scale加载逻辑，E38补编译了同环境Aper-token MATE算子库，并对真实target/MTP专家进行动态、静态两种短检查。无需再改模型代码即可单独实测该路径。

两组实际执行ELF均为 `/expt/artifacts/native_sg438_mccl_v1/bin/xllm`，SHA256 `6468707445ac26b74f6e8766dd20b13b5567e367a3d172d482aff777b0d09293`，源实现来自0130ef9a及其冻结构建清单。E38 d2a4b6d6提交的DeepEP接入原语还没有进入这个ELF，也没有用于本轮整模型服务。

- OFF：P/D均 `XLLM_DSV41_W4A8_STATIC_ACTIVATION=0`；原group128动态激活库。
- ON：仅D端 `XLLM_DSV41_W4A8_STATIC_ACTIVATION=1`；`XLLM_DSV41_MATE_W4A8_SO=/expt/builds/e38_static_mate_v1/lib/libdeepseek_v41_w4a8_static.so`，SHA256 `9a23ce3a6dfe2381e20573ae17643fba3d22760a79dd1e390bb5ab4531a36241`。P端仍动态。
- D端开关同时作用于target和3层draft专家：各rank加载本地专家partition的input_scale，gate/up取共同max，down独立取max；没有删除层或专家、没有更换模型文件。
- 其余维持E37：Attention TP8、路由EP8、共享专家TP8；target q4完整Graph，proposal3、静态draft，full draft Graph关闭。框架executor的eager字段为true，不应称整个模型eager。
- Torch/torch_musa2.9、MATE0.2.6、MUSA4.3.8及冻结SGLang镜像，沿用E37兼容环境。不能把相对更早E34的全部变化都归给本项静态量化。

## 2. 固定协议和实际结果

SJ9 P与SJ10 D，每台8卡；同W4AFP8-mt1017，C1、temperature0/top_p1、chat、ignore_eos，固定输入256含模板，实际提交输出256。EvalScope1.14.1，正式profiler OFF。快筛为1暖2次，正式另做1暖5次，快筛不混进正式均值。

|指标|OFF 动态|ON 静态|变化|
|---|---:|---:|---:|
|TTFT平均|296.523947 ms|331.764857 ms|增加35.240911 ms，仍低于990.04 ms|
|TPOT平均|30.083463 ms/token|27.843298 ms/token|降低7.4465%|
|TPOT样本标准差|0.254888 ms|0.338032 ms|均保留全部5次|
|E2E平均|7.967807 s|7.431806 s|降低6.7271%|
|输出token / latency之和|32.129292 token/s|34.446540 token/s|提高7.2123%|
|请求成功/失败|5/0|5/0|每次实际256/256|

工具窗口吞吐、QPS和百分位另见原始 `benchmark_summary.json`、`benchmark_percentile.json`，不与latency-sum吞吐混称。逐请求数字已重新从SQLite原始数据计算，与summary核对。工具响应库保存了SSE包，但本次没有usage包；通过实际response ID关联服务端 `sequence_terminal` 的 `prompt_tokens=256,generated_tokens=256,finish_reason=length` 再次交叉核验，未把SSE包数或文本重分词当作输出token数。

基本题前后均返回4.2、56。静态路径又完成一次2048/256冒烟（44.236828s，首有效内容1.717183s）；该请求包含4096容量Graph首次捕获，不是长输入正式性能结果。随后再次通过两道题和256/8短请求，检查长短图桶切换。正式2048/256的1暖5次仍留在最终验收。

## 3. 提速主要体现为验证轮数减少，不能说GEMM快了7%

|正式请求的每次工作量|OFF|ON|
|---|---:|---:|
|服务端真实accepted/proposed|119/408|130/375|
|真实接受率|29.1667%|34.6667%|
|proposal3轮数|136|125|
|首chunk之后wall time / 轮数|56.406493 ms|56.800328 ms|
|同组5次输出文本SHA种类|1|1|

静态与动态两组之间输出文本不同，但各组5次完全稳定。本轮没有把两种量化的输出等价作为晋级要求；现象已记录，不展开模型精度排查。

ON用更少的投机验证轮数交付同样256个token（136→125，减少8.09%），与总耗时下降相符；每轮wall time没有变短。其包含draft、verify、cache、主机及流式开销，且生成内容不同，不能叫隔离的GEMM时间。该证据支持“当前固定题目的有效工作量减少”，不支持宣称所有输入都获得同样收益，也不能据此认定模型精度已改善。

## 4. trace实际看见了什么

两组各P/D 8rank，共32份GPU trace和32份native C++ sidecar。独立抓实际256/8；核对Record function id、线程、实际ELF及Graph correlation。最后q4/kv265 step转换为Perfetto，634个native语义/ATen入口均来自实际回溯。

- 两边target每次Graph replay均4047个GPU节点；每步入口计数相同，没有“launch次数减少”的证据。
- MATE masked S4FP8核均80次，符号的activation group从`...,3u,128,false`变为`...,3u,-1,false`，与实际加载静态Aper-token库一致；这直接证明ON没有仍用动态库。
- 两边保留40次FP32专家custom归约、40次BF16 Attention custom归约、40次SWA fused read。DeepEP完整服务尚未启用。
- 最后target Graph设备跨度OFF51.924ms、ON52.925ms；只是重profiler下该次观察，不支持单轮GPU显著加速。独立256/8请求启用profiler约18.5s，关闭约0.48～0.74s，包含重新捕获和日志开销；不能拿图内区间占比直接当未profile性能的Amdahl上限。
- addr2line提示部分DWARF FORM不支持；函数名已解析并核验包含DeepseekV41，部分源码行不全，不能宣传每个frame都有完整行号。

![OFF实际CPU区间与GPU时间线](04_trace/静态激活_OFF/Perfetto图/真实CPU区间与GPU时间线.png)

![ON实际CPU区间与GPU时间线](04_trace/静态激活_ON/Perfetto图/真实CPU区间与GPU时间线.png)

![ON原生调用链计数火焰图](04_trace/静态激活_ON/Perfetto图/实际C++回溯_ENTRY_COUNT火焰图.png)

火焰图横向宽度表示ENTRY_COUNT，不是CPU时间或GPU时间；上方时间线才是实际时长。跨层Graph内部算子在GPU轨道回放，不应因CPU没有逐层ATen调用就认为层被跳过。看优化时必须同时看“单轮花多久”和“完成256token需要多少轮”。

## 5. 产物和恢复位置

- NFS根 `/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116`，容器挂载为`/expt`。
- OFF原始服务与客户端：`runs/p39_static_activation_off_v1`；ON：`runs/p39_static_activation_on_v1`。
- 5次正式原始DB：各run下 `formal_client/formal/deepseek-v4.1-flash-1017-w4afp8-mt/parallel_1_number_5/benchmark_data.db`；同目录含工具统计，上一层含HTML。
- 全16rank trace：各自 `profiling/p39_static_activation_{off,on}_v1/{sj9,sj10}`。
- 本地本目录 `原始证据_off/on`、两个tar.gz、`03_逐请求与trace分析.json`、`04_trace`。
- OFF归档SHA `4f8d980fbe471534f5d1310b6fa5edf930b7ef6682124bd625d1c62a78a0f746`；ON归档SHA `2c677eaf6f583c133ffbe30a7714a3501b5f6b3060f12061661b69ee989e3953`，均下载后再次校验。

两组共32个rank全部正常退出0，无SIGKILL。SJ9于06:45:08显存空闲；SJ10同秒观察仍有短暂释放延迟，06:45:36复查8卡0MiB、我方容器只剩shell/sleep，未做额外强制操作。

## 6. 晋级和后续

ON相对同ELFOFF提升7.21%，明显超过本项波动，作为后续组合控制；三轮最终复验仍未做。相对目标仍需降低TPOT约22.53%、提高输出速度约14.53%。旧E34快照与E37新环境控制保留，方便回退和归因。

E38的真实专家DeepEP链路已经短检查通过但尚未整模型启用。E40容量初算表明，全DP参数约71.95GiB/卡，不能直接叠加当前Graph开销。下一步优先核对冻结SGLang B非专家FP8 Linear及Graph缓冲组织，再补齐容量预算，接owner/idle、共享专家和LL。遇到容量依赖可先做有依据的前置实现，不把“清单做完”视为追平。
'''
(e/'05_结论与下一步.md').write_text(report)
(e/'01_代码与构建').mkdir(exist_ok=True)
for p in Path('/tmp').glob('match*e39*.py'):shutil.copy2(p,e/'01_代码与构建'/p.name)
for name in ['match_e39_cleanup_final.json','match_e39_on_post_long_result.json','match_e39_on_long_live.json','match_e39_after_on_preflight_sj9.json','match_e39_after_on_preflight_sj10.json','match_e39_spec_stats.json']:
 shutil.copy2(Path('/tmp')/name,e/'01_代码与构建'/name)
c=o/'00_总控与恢复';s=json.loads((c/'任务状态.json').read_text());s['current_experiment']='E39 completed; E40 DP capacity / FP8 dense prerequisite';s['status']='E39_PROMOTED_PHASE_BEST_E40_PREPARATION';s['previous_phase_best_new_run']=s['current_best_new_run'];s['previous_phase_champion_metrics']=s['phase_champion_metrics'];s['current_best_new_run']='p39_static_activation_on_v1';s['phase_champion_metrics']=on;s['e39_trace_validation']='ALL32_GPU_NATIVE_VALID_TARGET_GRAPH_KERNELS_CORRELATED';s['e39_cleanup']={'normal_exit_ranks':32,'sigkill':False,'sj9_free_at':'2026-11-17T06:45:08+08:00','sj10_free_at':'2026-11-17T06:45:36+08:00'};s['e39_long_result']={k:v for k,v in d['on']['long_smoke'].items() if k!='content'};s['e39_acceptance']={'off':[119,408],'on':[130,375],'output_text_changed':True,'scope':'fixed benchmark only, not accuracy validation'};s['next_action']='E40 inspect frozen SG B FP8 nonexpert linear effective kernels; complete DP/LL memory and Graph pool budget; implement needed low-cost prerequisite, short checks, full benchmark + trace on E39 best.';s['e40_parameter_budget_local']=str(o/'02_实验记录/实验_040/01_容量与源码依据/参数容量分组.json');s['updated_at']=datetime.datetime.now().astimezone().isoformat();s['last_goal_turn_classification']='progress: E39 OFF/ON formal, all-rank independent trace, long smoke, cleanup completed; E40 parameter budget exposes DP memory prerequisite';(c/'任务状态.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
with (c/'恢复记录.md').open('a') as f:f.write('\n## 2026-11-17 06:48 E39阶段晋级、E40预算推进\n\nE39 ON与OFF同ELF正式各5次成功。ON TPOT27.843298ms、TTFT331.764857ms、34.446540token/s，相比OFF输出提高7.2123%。真实accepted/proposed由119/408变为130/375，轮数136→125；同轮约56.4→56.8ms且图节点4047不变，不能把收益叫GEMM加速。32rank trace/sidecar已核验、6张真实Perfetto图片已渲染。2048/256单次含首次Graph捕获44.2368s，随后4.2/56和短请求通过。两组32rank全正常退出，SJ9/SJ10显存已确认释放，无KILL。累计控制改为E39 ON，保留E34/E37。E40 CPU文件头预算约71.95GiB/卡仅参数，当前服务GMI72.62GiB再加DP参数差额将超80GiB；需优先核对SG非专家FP8 Linear或Graph池，不盲目启动DP。E40文档明确未完成完整峰值预算。下一步见状态next_action，goal仍未达标。\n')
(o/'03_累计最佳/E39_静态激活阶段最佳.md').write_text('# E39阶段累计最佳\n\n详见[实验39](../02_实验记录/实验_039/05_结论与下一步.md)。\n\n同W4AFP8-mt、1P1D每台8卡、实际256/256、C1，target q4完整Graph/proposal3、D端静态专家激活。TPOT27.843298ms、TTFT331.764857ms、输出34.446540token/s。每侧1暖5次，独立trace、长输入冒烟和基本题通过；未做最终三轮和长输入正式5次。\n\n实际ELF6468707445ac26b74f6e8766dd20b13b5567e367a3d172d482aff777b0d09293；静态AOT9a23ce3a6dfe2381e20573ae17643fba3d22760a79dd1e390bb5ab4531a36241。run=p39_static_activation_on_v1。\n\n仅通过基础功能检查，未完成模型精度验收。未达到最终追平门槛。\n')
print('E39 report, champion, state and recovery saved')
