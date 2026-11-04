# 回归测试结果与覆盖边界

原始日志已从共享盘下载到 `证据/component_regression_evidence.tar.gz`，SHA256 `1ddfced5ef7636af0bec1c6e4a06464ade8c3b615e715d8d9085d2a7c9eec1ba`。这些是已经完成的测试结果，不是本次整理时重新运行。

| 层次 | 结果 | 关键内容 | 包内日志 |
|---|---|---|---|
| MATE KDA | 5通过，75.07秒 | 长度1/17/65/129/257，对独立高精度递推参考、输入不被意外修改 | logs/components_v1.log |
| RMSNorm/mHC pre | 合计5通过，52.49秒 | RMS宽度128/512/4096；mHC token数1/17、hidden4096 | logs/components_v2.log |
| Folly/bthread | 3通过 | 启动与idle回收兼容设置 | logs/test_folly_v1.log |
| 缓存布局 | 23通过 | MUSA混合KV/线性状态布局、TP分片语义 | logs/test_cache_layout_v2.log |
| FP8离线载入 | 10通过，1.98秒 | 分片后反量化及非对齐分片处理 | logs/test_fp8_offline_v1.log |
| XServiceClient生命周期/ABI | 5通过，31毫秒 | 包含跨翻译单元读取初始化状态的回归 | logs/test_lifecycle_abi_v4.log |
| Python模型回归 | 初次347通过、115跳过、1失败 | 失败是测试夹具的父模块绑定问题 | logs/test_python_model_regressions_v1.log |
| 上述失败项修复后定向复测 | 1通过，0.07秒 | 修复测试夹具后单独复测 | logs/test_kpool_fixture_binding_v2.log |

不能将最后两行合写成“一次运行348项全部通过”；没有在修复后重跑整个集合。115项跳过也不能算通过，更不能据此证明graph、CUDA/NPU等所有后端可用。

## 各层证据分别证明什么

算子数值测试检查局部计算是否符合参考；缓存/生命周期测试检查宿主框架的关键约束；随机小模型的单机TP8与跨机1P1D对照检查请求与状态交接；真实GLM基础问答证明完整权重链路能回答简单问题；MathQA对分才是在固定题集上检查任务级效果。任何一层都不能替代其他层。

当前跨机随机fixture的4/17/129输入均与单机输出一致，P/D共16个rank正常退出码0，详见 `证据/tiny_validation_v4_evidence.tar.gz`。真实完整模型3道基础问答均正确，详见 `证据/full_smoke_3questions.tar.gz`。双边MathQA前10题均10/10；官方额度已补充，官方100题已完成93/100；本地已按用户要求停止扩展，冻结65个初始HTTP200响应，其中60题匹配标签；正式性能未运行，阶段结果以paired_protocol_v2/stopped_snapshot为准。
