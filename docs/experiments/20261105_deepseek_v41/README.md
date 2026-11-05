# DeepSeek-V4.1-Flash 0909：初始 MUSA 适配与单机八卡预检

**这是阶段版本：已完成单机预检，双机 1P1D 与正式性能尚未完成。** 起点为 `1103-merged-dev@0cfd82700e164e8aa7951064b652925dd6b24d18`，个人分支 `xiaopangDeepseeekV41Flash`。详细实现及边界见 [适配说明](适配说明.md)。

2026-11-05 在 dev7 八张 S5000 上使用新编译 xLLM、我方 Python 源码、1102-dev 镜像依赖完成加载。attention TP8，专家 EP8/MoE-TP1，eager，C1，无 MTP、无 prefix cache。保留原始 MXFP4 专家存储，活跃专家在 GPU 临时展开为 BF16 计算；没有宣称接入 MATE MegaMoE 或完成官方精度对齐。

| 验证 | 结果 | 证据 |
|---|---|---|
| 1.3 + 2.9 | 4.2；21 输入 / 4 生成 token | raw/runs/load_precheck_v2/basic_questions |
| 7 × 8 | 56；17 输入 / 2 生成 token | 同上 |
| 256/8、2048/8 输入长度及 chunking | 两次实际 usage 符合；输出可读英文开头 | raw/runs/load_precheck_v2/prompt_length_smoke |
| 正常退出 | 8 rank 全 0，容器 0，八卡显存全 0 MiB | raw/runs/load_precheck_v2/shutdown_result.json |
| 原生定向回归 | 6 组共 40 用例通过 | raw/evidence/native_validation_v1 |
| 综合缓存回归 | 189/264 通过；75 失败保留；V41 名称用例 42/43 通过 | raw/evidence/native_cache_cases_v1 |
| GPU / EP8 小形状验证 | 范围和误差见原始 JSON；不是完整模型精度 | raw/evidence |

综合缓存失败中 70 个触发旧 V4 C4/C128 容量断言，另有 5 个旧 MTP/LINEAR 等路径失败。唯一带 V41 名称的失败发生在旧 V4 对照组构造。完整列表和日志保留，不以删测试或放宽断言换取通过。

后续计划：等待 dev4–7 中两台八卡全空闲；启动真正两机 P/D，各 TP8、专家 EP8；确认缓存/Engram history 实际传输与两题正确；使用原版 EvalScope 1.14.1 在 C1 下测 256/256、2048/256，各预热 1 次和正式 5 次；核对原始 usage，再报告 TTFT/TPOT/端到端输出 token/s。单机短输出结果不算该性能基线。

NFS 完整工作目录：`/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261105跑通PD分离/`。原始 0909 模型只读使用。1017 下载是独立资产任务，不混入本轮测量。私有 MATE 源码、镜像、权重和凭据均不随本实验记录提交。
