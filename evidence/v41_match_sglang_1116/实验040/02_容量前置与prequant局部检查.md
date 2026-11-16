# E40阶段证据：DP容量前置与SG prequant路径

当前完成容量静态推算、真实权重的SG prequant算子微基准与动态输入Graph回放。正在补原生C++ AOT入口；尚未接进模型，不是新性能冠军，也不是DP部署已完成。

## 容量约束

保留列出的Q-A/Q-B/KV/O-B/共享专家FP8权重而不保留BF16副本，可节省5.123949GiB/rank。完整DP参数估计66.826231GiB；这仍未包含Graph/cache/加载峰值。不得据此直接启动完整DP。当前图池已经共享，已从候选中排除重复实施“共享图池”。

## 本次与E19/E25有何不同

E19/E25是旧环境、合成权重、每个N tile重复量化的fused quant GEMM；本次采用冻结SG438镜像及真实1017 W4AFP8 layer0权重，激活量化仅执行一次，再运行SG prequant GEMM。当前native activation floor=1e-4，因此调用SG量化函数时显式eps=1e-4，而不是默认1e-10；未修改模型资产。

SG B实配启用TP1_PREQUANT等开关；完整QB/OB/共享矩阵均可命中。q4是我方当前最佳需要支持的形状，q6是SG B形状；本次显式扩展检查q1/q4/q6，不宣称SG默认q4一定走这条优化分支。

## 真实权重小检查

6组矩阵、rows1/4/6、普通/微小输入、默认/非默认stream，共72项通过，相对BF16载体reference最大RMS=0.000330419，小于预设1%。这些局部检查不代表模型精度验收。另对q4/q6捕获后更换3组input值，共36次回放通过，最大RMS0.0001999805。

每项Graph event微基准30个调用/图，3次回放组成一个样本，5个样本取中位数；先控制、候选、再反向控制。没有正式整模型HTTP负载同时运行。单位微秒/投影调用：

|投影|行数|BF16载体＋QDQ|SG先量化＋FP8GEMM|反向控制|
|---|---:|---:|---:|---:|
|qa|4|29.004|22.275|29.019|
|qa|6|29.672|22.517|30.014|
|kv|4|20.248|18.878|20.620|
|kv|6|20.929|20.094|21.010|
|qb_h64|4|82.394|65.780|92.110|
|qb_h64|6|87.218|75.219|84.716|
|ob_h64|4|67.161|62.076|67.401|
|ob_h64|6|67.482|64.024|67.324|
|shared_gate_up|4|56.240|48.067|56.690|
|shared_gate_up|6|55.665|43.918|55.102|
|shared_down|4|27.763|30.748|28.145|
|shared_down|6|27.687|31.412|27.557|

QA与完整QB/OB等值得接成原生前置；shared-down较慢，不能将全表概括成所有FP8矩阵提速。QB样本波动较大，原样保留全部数据。shared gate/up在此两路均采用合并矩阵以比较量化/GEMM，本模型当前q4 shared仍可能是两次独立投影；不能把本表当作现网模块时长。

## 实现中的下一步

1. 原生C++入口调用离线编译的quant/GEMM/reduce，检查其与同一BF16载体基准的数值、非默认stream和动态输入Graph回放；导出成功不是运行正确。
2. 接入checkpoint原始FP8/scale的按TP或完整矩阵加载；不重复持有BF16载体。P端继续原路径，D端按显式支持shape选路，不能异常后偷偷fallback。
3. 先在隔离完整模型候选验证，更新所有rank静态/缓存/Graph/加载峰值容量，真正接E38 LL owner/idle和DP Attention。TP8 head/Engram、共享/路由归约分别核验。
4. 基础题、正式1暖5和独立trace完成后才决定是否晋级；当前继续保留E39 proposal3。

NFS微基准：`/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/diagnostics/e40_prequant_real_weights_v1`；Graph动态输入：同diagnostics下`e40_prequant_graph_refresh_v1`。原始结果、日志、脚本与容量JSON保存在本地原始证据_prequant_v1及原始证据_graph_refresh_v1。C++与AOT新产物在builds/e40_prequant_native_v1和artifacts/e40_prequant_aot_v1，未完成原生GPU验收前不部署。

**仅通过基础功能检查，未完成模型精度验收。** 本节局部数值通过也不代表整模型通过基础题，当前无新的端到端提升数值。


## 原生 C++ 接入的第一轮结果

已编译并在 GPU 上执行原生 C++ launcher：72 项真实权重检查通过，最大相对 RMS 0.000330419；36 项不同输入 Graph 回放检查通过，最大相对 RMS 0.000199980。原始记录在 `原始证据_native_checks_v1/`，归档 SHA 为 `ff78fc9ce6ca8302ac12962d41466245192b8444493084460d10adec1441a0c4`。

这不代表速度通过。第一轮 AOT 在若干形状上慢于 SG 的 JIT，例如 q6 的完整 O-B 投影，BF16 控制约67.67us、原生prequant约121.75us。核对生成的TTIR，SG有16字节指针对齐特化，原生AOT没有；正在隔离版本补齐，并由C++校验外部输入/权重/scale指针的对齐，不能只根据contiguous就作此保证。最终是否解释差距以新实测为准。

编译第一轮链接误写成`-lmudart`而失败，已修正为本环境的`-lmusart`，保留失败日志。第二次编译补齐容器内`MUSAMAPPING_PATH`；没有修改宿主驱动。
