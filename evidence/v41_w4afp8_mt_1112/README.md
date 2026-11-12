> 后续状态更新（2026-11-12）：本文件保留第一阶段基线记录；小范围优化及两组复测已完成，见 [StageI图文报告](optimization_stage_i/README.md)，H200参考与显存释放见 [NVIDIA记录](nvidia_h200_reference/H200首轮原版1017同Token测速说明.md)。H20 GPU任务已由用户延期。

# 我用交付版 W4AFP8-mt 权重完成的原生 C++ V4.1 基线

本次基于共同仓库`1103-merged-dev`的`0cfd82700e164e8aa7951064b652925dd6b24d18`，在个人分支`xiaopangV41FlashCppFromSglangtoXllm`接入V4.1文本推理。共同分支地址：https://github.com/fay85/xllm-musa/tree/xiaopangV41FlashCppFromSglangtoXllm 。本阶段没有创建MR/PR。这里是第一阶段的算子与完整测速记录，后续基于trace的一处小范围优化会单独提交。

## 这次实际运行的实现

模型主体是C++的`DeepseekV41NativeCausalLM → DeepseekV41TextModel → 40个Decoder`，没有在线调用Python模型forward。Python用于离线AOT导出/构建、分词器sidecar、数值对照、启动与测试客户端。sj9做P、sj10做D，各8个独立MUSA进程，Attention TP8、路由专家EP8（每rank48/384个专家）、共享专家复制后仅加一次。两端各8个rank都确认完整加载40层，D每个rank有13条非空Mooncake拉取记录；不能用regions=0的成功日志代替数据实际传输。

权重是`/data/nfs_shared/models/DeepSeek-V4.1-Flash-1017-W4AFP8-mt`，ModelScope固定commit `a41b72782c27dbd505c84c13fcefaabb1fe2f273`，99文件/91权重分片，510,320,508,387字节，143,317个张量，整文件SHA与索引已验收。它的专家编码实际为signed INT4打包I8、group128和FP32 scale；不要把它当作原版E2M1 FP4/group32直接调用。激活仍需按各模块语义处理，换专家权重没有消除Dense/Indexer的全部量化问题。

参考来源：家明V4的`f8dd0663279e9085365ade309d9823b7b438c4e6`；京东交付SGLang镜像内实际源码`7c97bb0e6d8b0f8fd41cd797a77b54f7162c79f6`及MATE实现；MooreThreads官方W4AFP8-mt模型。不是只按交付指南中的启动命令推测算子覆盖。

## 已接入什么，还保留了什么

- 专家路径：C++负责路由、分发、FP8激活量化、clamp/SwiGLU、结果合并与EP归约，大GEMM接交付版MATE S4FP8。该MATE内核内部将A转FP16并以INT4 B/F32累加执行，不声称是S5000原生FP4×FP8 TensorCore峰值实现，也不是原版专家逐个解量化为BF16的旧路径。
- mHC：保留V4.1 shifted pre_mix跨子层传递与20次Sinkhorn，接MATE prenorm；不能照搬V4的自身pre_mix消去规则。
- CSA2：接H8/D512 MATE稀疏Attention；修复无效pick读取未使用NaN KV行的问题，保留NaN/全无效pick定向回归。Indexer score、稳定TopK512、RoPE/FP4量化融合、压缩KV的E4 scale语义与C2分块状态均完成对应测试。
- Engram：离线生成目标Tokenizer对应hash sidecar，在线hash/history、GPU行收集、注入和跨TP汇合由C++/MUSA执行；两层Engram都参与计算，未以CPU算模型或跳过模块换取跑通。
- 小batch投影复用1103已有WO_A M1实现；RMSNorm、C2 pair pooling、Engram gather/hash等均有真实GPU验证。Dense block32候选虽数值通过，但在实际TP8 M1形状比现有路径慢，所以未为“接了新算子”而启用更慢方案；当前Dense仍是GPU QDQ + BF16 MUBLAS。
- 保留完整chunked prefill。交付版可选CED/late-SWA bounded replay会裁剪后层prefill行，不能在没有语义回归时直接等同完整prefill。本阶段没有启用Graph、MTP、视觉、DeepEP/all-to-all，也不宣称已经覆盖通用并发serving。更广的metadata/缓存融合和并行策略是后续优化范围。

逐模块文件、数值边界和交付版差异另见`operator_coverage.md`。全部数学工作仍在MUSA GPU；少量host长度/页表/调度元数据不等于CPU模型forward。

## 正式性能：同一配置下两组都完成

运行模型ID `deepseek-v4.1-flash-1017-w4afp8-mt`；C1，TP8/EP8各一台，eager，chunk/batch256，maxseq1，cache1GiB，block128，prefix cache关、overlap关、MTP0。未修改的EvalScope **1.14.1 perf**，流式接口，temperature0/top_p1、显式无思考模板，ignore_eos=true。每组1次暖机+5次正式请求，暖机不计均值；全部请求由服务usage证实实际输入256或2048、输出256，不以max_tokens代替实际输出。

|输入/输出|正式成功数|TTFT ms|TPOT ms/token|端到端 s|官方输出吞吐 token/s|官方QPS|
|---|---:|---:|---:|---:|---:|---:|
|256/256|5/5|395.10|205.17|52.7143|4.8563|0.0190|
|2048/256|5/5|2413.25|207.13|55.2314|4.6350|0.0181|

终态为`PASS_ALL_FOUR_PHASES_EXACT_NATIVE_TOKEN_COUNTS`，北京时间2026-11-12 09:22:52完成。基础问答1.3+2.9=4.2、7×8=56通过；另有实际2048输入/2输出的长题校验。此为基本功能门槛，没有做MathQA100，不宣称完整精度或榜单分数对齐。

QPS和吞吐沿用官方组时长口径，官方JSON有四舍五入。TPOT按实际输出token计，SSE分块产生的ITL与每token耗时可能不同。官方摘要里出现的`Spec. Accept Rate`衍生栏不作为MTP启用证据，实际冻结参数明确`num_speculative_tokens=0`。

### 必须随数字一起说明的配置边界

这组完整基线开启了`XLLM_DSV41_DIAGNOSTIC_PROGRESS=1`逐层CPU日志和`TORCH_MCCL_TRACE_BUFFER_SIZE=65536`，以及通信诊断配置；**profiler未开启**。不能把它冒称为所有默认配置下都稳定。先前F/G在无该逐层日志、记录缓冲区2000的配置下，短组5次通过，但随后2048暖机出现RPC300秒和MCCL600秒超时。G将同成员TP/EP归约复用同一通信组，改善短组数据，但没有单独解决长组故障。目前没有隔离证明究竟是哪一个观测配置影响了时序，不能把这次成功归因给“日志修好了死锁”。这些失败另留账，后续需逐一消除变量。

还有一轮H诊断在全部请求完成后，我用GDB暂停P主rank采栈，导致3秒租约过期/实例被剔除；其后503是观测引入的服务状态失效，已归档并正常重建本次H正式v2。正式测速期间没有GDB暂停、没有profiler、没有其他GPU测试。全程未改驱动、未重启服务器，未停止他人任务。

## 环境、产物与原始证据

- 基础镜像`registry.mthreads.com/presale/devtech/xllm:1102-dev`，image SHA `09b8f5f4a467fc73332dee57b36a2e5610ba3264357dfc87f103cf7bd7430be3`。Torch/TorchMUSA2.9.1，SDK5.1.0，基底MATE0.2.7；交付版特殊MATE/TVMFFI/TileLang仅按任务路径冻结加载，没有全局升级。
- 实测主程序SHA256 `5dec7bad2087d42fae7c0eb7a23a2ffe5ee64555310b7793edd84047a7508799`。运行包payload32项、构建时修改源码166项各自SHA在原始包的`artifacts/native_stage_h/{manifest,source_file_sha256}.json`。后加的离线读图工具不参与这次二进制，不能用整个提交SHA替代编译输入清单。
- 容器：`xiaopang1112_v41_prefill_sj9`、`xiaopang1112_v41_decode_sj10`。NFS任务根`/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261112_sglang_mt_cpp`，原始性能在`performance/stage_h_progress_logging_v2/`，运行参数和16rank日志在`runs/stage_h_formal_v2/`。
- 本目录`baseline_stage_h/baseline_stage_h_raw_logs.tar.gz`包含完整官方性能原始数据库/请求与响应、暖机和正式摘要、gate、16rank日志与配置及产物清单。100个成员逐个SHA核对；压缩包SHA **bef6175eed1b6a55ed56fc51404ed362d0751f7b42ab5ff213c506d0deb8d8d0**，691,253字节。`archive_manifest.json`可逐文件核对，`summary.json`便于速读。
- 运行环境实际驱动/设备名有边界：sj9/sj10的gmi报告X10000、3.3.8-server、8×80GiB；不把dev6的MTT S5000/5.1驱动信息拿来充当sj的硬件实测，也不直接从NVIDIA token/s乘0.6推导MUSA的60%SOL。

## 后续一步

先固化这份可复查基线；随后抓暖态真实C++栈与GPU trace，只选一处少文件改动，做数值回归与同条件两组复测及真实Perfetto配图，再提交第二个commit。现有16rank冷诊断trace已证明原生栈采集链，但不会代替当前基线的优化前后trace。并行准备H200原版1017的同输入对照；201仍有外部GPU任务，H20对照等待资源，不停止他人容器。
