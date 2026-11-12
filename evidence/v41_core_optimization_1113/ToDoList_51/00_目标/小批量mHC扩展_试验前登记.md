# T03/T12：已验证的mHC归约融合扩展到2～8行（尚未测试或启用）

实际T14 trace显示，现有融合只在q1生效；q6 verify与q4 replay走原有分散归约。当前融合operator的40个基本用例与6个真实MATE权重用例已覆盖rows17/128，但模型选择条件仅rows==1。这也会使后续B2/B4丢掉已接受的mHC局部融合。

新增独立默认关闭开关XLLM_DSV41_MHC_SMALL_BATCH_FUSION，仅在原有mHC融合已启用、2≤rows≤8时选择同一现有融合operator；q1仍保留原有已接受路径，P端默认不启用mHC融合。先做2/4/6/8行的实际MATE及HyperConnection相关短回归，再在相同B2场景只切换这个开关，完成两组正式性能与真实trace。若没有净收益不晋级；不把它与首次启用B2的收益混在一起。

这不是已完成的性能改进，不跳过Gate、chunk、CED、prefix及批处理基本任务。它只使用dev6 CPU编译，GPU仍只有当前SJ9/SJ10试验。仅模型基础功能验收，不开展模型精度评测。
