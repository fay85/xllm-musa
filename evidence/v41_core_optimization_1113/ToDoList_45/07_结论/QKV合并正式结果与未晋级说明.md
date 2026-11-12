# QKV投影合并：执行路径确认，收益混合，不晋级

在E38累计最优、同c608 ELF上，只开启D端XLLM_DSV41_DECODE_PACKED_QKV。共享专家packed保持0，其他已接受开关保留。两道计算题通过；256/256与2048/256各1暖5正式，10次请求成功且实际token数量正确；16rank正常退出，无SIGKILL。

|输入/输出|状态|TTFT ms|TPOT ms/token|端到端s|输出token/s|
|---|---|---:|---:|---:|---:|
|256/256|before|395.42|67.15|17.5186|14.6130|
|256/256|QKV packed|406.58|67.55|17.6319|14.5192|
|2048/256|before|2672.03|66.19|19.5510|13.0940|
|2048/256|QKV packed|2679.91|64.32|19.0802|13.4170|

短TPOT上升0.60%、长下降2.84%；样本标准差分别约2.30/1.21ms。该轮没有证明两组都稳定获益，所以不进入默认最优，也不称为确定回退。保留实现及结果，不挑长输入来宣称整体提升。

32rank的64个原始trace/stack文件均完成SHA与实际ELF PC符号化。八个D rank每步mm349→309，完整7step GPU kernel减少560，即80/step；copy_837、cat82保持一致。模型加载时将共享输入的wq_a/wkv沿输出维打包，q1一次linear后narrow得到两路输出。省去重复linear及相关中间分配；新narrow/slice各增加80次。真实调用减少不等于全局收益一定超过波动，矩阵shape变化和CPU视图开销也存在；没有无profiler关键路径数据，不能强行把差异归因给其中一个。

正式SQLite/SSE/工具报告/角色参数/服务日志位于04_性能/t06_packed_qkv_cumulative_v1_raw。原始完整trace路径和SHA见05_trace/processed_trace_bundle_v1.receipt.json；全8D计数见05_trace/all8_packed_qkv_effect.json。图由真实pftrace渲染，UI加载失败的截图目录另存FAILED_UI，不为截图重跑模型。ENTRY_COUNT为实际C++语义入口次数，不是CPU时间。

仅通过基础功能检查，未完成模型精度验收。

图像查看补充：after的第一张火焰图截图在profile数据尚未绘制完时拍下，时间线可见但底部尚空，不能据此说没有native调用。`AFTER_PACKED_QKV_D0/完整实际C++调用链_ENTRY_COUNT.png`已实际查看，包含完整630-entry调用链；配合两侧时间线与all8计数读图。UI初次加载失败已另存FAILED_UI，原始trace与正式测试不重跑。
