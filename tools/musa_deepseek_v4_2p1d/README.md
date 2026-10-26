# DeepSeek-V4 MUSA 2P1D + PCP

完整构建、三机启动、停止、性能和 GSM8K 命令见
[操作文档](../../docs/DSV4_2P1D_PCP_BUILD_START_PERF_ACCURACY.md)。

- `build_site.sh prefill|decode`：90/94 `wjm_xllm1008` 的完整环境和构建命令；依赖现有 NFS 工具与角色 wrapper，不是通用安装脚本。
- `bench.py`：固定 prefix token 复用比例 0.8，输入支持 20K/40K，输出固定 100，并发上限 1～16。默认 40K、并发 1～5。
- `gsm8k.py prepare|execute|score`：冻结 seed=20261026 的 100 道 GSM8K，分别测试短输入和 40K 输入；推理不加载 gold，无重试。

性能每档两轮、每轮 32 条，顺序升序再降序；另有两个冷 primer、两个热探针。必须使用新目录，准备后执行阶段不允许覆盖已冻结的参数。20K/40K 分别要求 16000/32000 cached tokens。

`summary.json` 同时保留请求通过状态、TTFT、TPOT、聚合吞吐和缓存统计。失败即停止，原始 SSE 和响应保留。计数工作负载通过不能替代 GSM8K 精度检查；仅 `--prepare-only` 通过也不等于已经完成性能测量。

启动 profile 显式记录 `XLLM_DSV4_GRAPH_JOINT_COMPRESSOR_PROJECTION`。本次修复保持请求局部 `Linear::forward()` 路径；对比时固定该开关及 P/D 二进制，重新验证精度与完整性能矩阵。P/D 必须分别配套自己的 Mooncake `libasio.so`，不要 preload 旧的外置 custom-allreduce 库。
