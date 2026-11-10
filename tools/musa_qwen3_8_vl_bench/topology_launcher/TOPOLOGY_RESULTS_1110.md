# 优化版本多拓扑结果（2026-11-10）

本轮全部配置已完成测试或确认当前实现限制。5组取得有效性能，2组内部DP在预热失败，2组聚合CP受当前二进制门禁限制。

## 统一口径

- 主机 `s5000_dev92`，容器 `albert_xllm_92`。
- 模型 `Qwen3.8-27B-FP8`；同一份优化engine，MTP K=3。
- 每组 C=8，正式60条fresh请求；总输入22752 tokens，含5张854×480 JPEG，输出100 tokens。
- 每个正式请求独立核验cached=20480、prompt=22752，即90.014%的token KV命中。
- 每组fresh stack；16次tiny warm、28次长输入warm、240次closed-loop full-output warm、64次同步full-output warm。正式60条没有在warm阶段重复提交。
- TTFT为客户端首次非空content时间；TPOT为(最后输出时间−首个输出时间)/(输出tokens−1)。QPS为正式阶段请求吞吐。
- 5组有效结果均60/60实际输出100、逐RID MTP计数为正；所有rank正式阶段capture=0、error=0。
- 本轮每配置一次60条正式请求；没有置信区间，结果仅代表C8/该负载。

## 有效性能

延迟均为ms；目标为TTFT P90<1000且TPOT P90<20。

| 拓扑 | 卡数 | QPS | TTFT均值 | TTFT P50 | P90 | P95 | TPOT均值 | TPOT P50 | P90 | P95 | MTP接受率 | P90目标 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| TP2（聚合） | 2 | 1.414 | 985.61 | 873.16 | 1998.47 | 2307.14 | 44.39 | 45.25 | 58.61 | 63.08 | 59.09% | 未达标 |
| TP4（聚合） | 4 | 2.022 | 695.59 | 577.31 | 1267.22 | 1522.75 | 30.74 | 31.77 | 39.20 | 43.43 | 58.71% | 未达标 |
| 4P4D（均TP1） | 8 | 3.888 | 496.43 | 453.44 | 625.95 | 831.25 | 14.52 | 14.36 | 15.97 | 16.70 | 58.04% | 达标 |
| 6P2D（均TP1） | 8 | 3.521 | 484.41 | 455.17 | 584.63 | 636.68 | 16.82 | 16.83 | 19.06 | 19.90 | 57.51% | 达标 |
| 2P×CP2 + 4D×TP1（补充PD组） | 8 | 4.115 | 399.91 | 331.35 | 578.58 | 835.00 | 14.47 | 14.38 | 15.71 | 16.43 | 58.42% | 达标 |

## 无有效性能的配置

| 配置 | 计划卡数 | 状态及原因 |
|---|---:|---|
| TP2×DP2（world=4, dp=2, cp=1） | 4 | 首tiny warm时DP1 ranks2/3触发 attention_metadata_builder.cpp:493 device.has_value() CHECK；没有正式结果 |
| DP2（world=2, dp=2, cp=1） | 2 | 首tiny warm时rank1触发同一CHECK；没有正式结果 |
| 聚合CP2 | 2 | 当前VLM MUSA CP2门禁要求PD角色，聚合模式不受支持；未绕过门禁 |
| 聚合CP2×TP2 | 4 | 当前CP2路径另要求world=cp且dp=1，尚不支持CP与TP组合；未绕过门禁 |

内部DP失败是空DP分支图捕获时attention builder缺失device的具体实现问题，不代表所有DP部署不可行。4P4D/6P2D是多个独立TP1服务，不能等同为上述内部DP2配置。

## 本轮比较及边界

- 补充的合法PD CP2组为2个Prefill逻辑实例×CP2（共4卡），加4个Decode×TP1（共4卡），总计8卡。它不是2卡聚合CP2，也不是CP2×TP2。
- 该组本轮QPS4.115，对比4P4D的3.888高5.8%；TTFT P50从453.44降至331.35ms，P90从625.95降至578.58ms。
- 4P4D相较6P2D吞吐更高、TPOT更低；6P2D的TTFT P90略低。两者均满足P90目标。
- TP2实际只用2卡、TP4实际只用4卡；PD组用8卡。总QPS比较包含资源数量差异。
- TP2/TP4为单聚合实例，不是历史8卡的4×TP2/2×TP4，不能用这些数直接判定性能回退。
- 本轮没有修改引擎/Service源码或重新编译。已恢复原优化4P4D服务，重启后缓存为冷态；性能表来自恢复前各自完成的正式阶段。

## 产物与复现

容器总目录：`/workspace/topology_sweep_1110`。
启动器及计划：`launcher/start_topology.sh`、`launcher/plans/*.json`。
统一测试：`bench_topology.py`；控制器 `run_matrix.py` 与 `run_cp2_reference.py`。
原始结果：`results/20261110_061156_*_c8`及`results/extra_pd_2p_cp2_4d_c8`。
有效启动命令、GPU/rank映射、身份记录及日志：对应`runs/`目录下`effective_commands.json`、`effective_plan.json`、`identities.json`、`*_r*.log`。
本地独立逐请求复核：`matrix_analysis/*/qualification.json`与`per_request.csv`；CSV含更完整分位。

## 二进制及恢复状态

Engine MD5：`ac2ff65a1e146022838775ac7bea368c`。
Service MD5：`4c55f164541dcc6564d38f02db86d902`。
恢复运行目录：`/workspace/ttft_delivery_1110/release/xllm_ttft_fix_1110/work/runs/20261110_064722.fat7YA`。
全局config SHA256保持`b5354c2672af18e8ec4f87742b0e30c02575176d3e09acc7551b10ad2f97171f`。
