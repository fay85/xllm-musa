# T02b：融合 q1 SWA 读取、候选索引与写回位置整理

**接受为累计速度基线。仅通过基础功能检查，未完成模型精度验收。**

## 同一二进制的正式结果

SJ9为P、SJ10为D，各8卡、TP8/EP8，eager、C1、chunk256。沿用D端TP/EP custom及压缩缓存复用。本项只改变D端`XLLM_DSV41_DECODE_FUSED_SWA_READ`。P端两模式固定启用显式FFI同步（见ToDoList_12），避免把P设置变化混进D融合的归因。ELF SHA为`3adb9a58adbc5830a8104a27a1dc62933eee49f11b4b44b9aba092fb8989aaf9`。

|输入/输出|SWA读取|TTFT ms|TPOT ms/token|E2E s|输出token/s|QPS|
|---|---|---:|---:|---:|---:|---:|
|256/256|off|432.990|143.549|37.037877|6.911843|0.026999|
|256/256|fused|458.994|125.906|32.565093|7.861178|0.030708|
|2048/256|off|2600.317|143.735|39.252723|6.521841|0.025476|
|2048/256|fused|2630.821|126.234|34.820559|7.351979|0.028719|

两组TPOT分别下降12.29%/12.18%，端到端输出吞吐提高13.73%/12.73%。全部20个正式请求成功、实际输入256/2048且输出256、同组请求哈希一致；每组1暖+5正式，速度阶段关闭profiler，trace单独采集。

**TTFT没有改善：** 256输入增加26.00ms（6.01%），2048增加30.50ms（1.17%）。所有样本保留，图上能看到短输入约200ms的分组波动；没有证据把这部分差值归因于SWA内核。本轮按TPOT和整体输出速度接受，不宣称TTFT也提升。P显式同步自身尚不能被称为已修复间歇性挂起根因。

![全部正式样本](../06_图/正式性能全部样本.png)

## 为什么想到这样改

来源于交付SGLang对页表/窗口metadata统一准备的思路，以及此前真实trace：同一个q1 token在40层中反复以arange、整除/取余、index_select、mask、cat整理SWA。当前MATE Sparse Attention已经接入，不必重写Attention数学计算；缺口位于xLLM的读取及metadata包装。缓存复用完成后，这部分仍有明显的大量小操作和launch。

新增MUSA kernel一次读取GPU positions与物理页表，直接构造history+current+compressed候选行、有效picks以及物理写回slot。计算结果继续交给同一个已验证的MATE Sparse Attention，最后沿用原单token scatter；保持先读后写。只处理q1，Prefill旧路径不变；没有跳过Attention或改GEMM精度，没有CPU模型计算fallback。

实现位于`deepseek_v41_swa_decode_read.h/.cu`（设备kernel）、`paged_attention.h`（执行顺序与选择）、`attention.h`（传positions/开关）、`deepseek_v41_native.h`（原生配置）。测试绑定和独立验证脚本调用实际C++/GPU实现。

## 从实际trace读到的变化

前后各16rank的文件大小/SHA、40层、真实RecordFunction/C++入口回溯关联均通过。每一个D rank都出现40次SWA_decode_read_metadata；index_select均259→139，变化不是只挑rank0。

|D0同形状q=1/kv=263窗口|OFF|FUSED|
|---|---:|---:|
|CPU index_select入口|259|139|
|CPU arange入口|260|20|
|CPU where入口|374|138|
|CPU item入口|1573|461|
|窗口内完整GPU kernel|5222|3757|

新增SWA kernel在选中CPU窗口内有39次完整事件，最后一个异步kernel可能跨窗口边界；CPU实际有40层/40次调用。不能把39解释为只执行39层，也不能把窗口kernel数当作整个异步step的绝对总数。`aten::item`入口也不等于GPU到CPU同步次数。原始窗口外事件仍保留在完整NFS trace中。

![实际ATen调用数](../06_图/实际ATen调用次数.png)

![之前实际时间线](../06_图/BEFORE_D0/真实CPU区间与GPU时间线.png)

![之后实际时间线](../06_图/AFTER_D0/真实CPU区间与GPU时间线.png)

两张时间线自动缩放不同：重profiler选中CPU区间约467.98→345.85ms，不应当直接拿它们替代正式的143.55→125.91ms/token。应先看相同层/相同语义下小ATen操作及对应kernel数量减少，再用profiler-off的正式数据判断收益。入口COUNT火焰图510→550，是因为每层新增一个可见SWA整理范围；COUNT增加并不表示变慢。

![实际C++入口COUNT](../06_图/AFTER_D0/完整实际C++调用链_ENTRY_COUNT.png)

## 基本检查、收益上限与边界

352个真实读取/索引用例及44个完整Attention检查与旧路径严格相等；包括I32/I64页表、宽128/512、空或有压缩行、跨页/环形位置、负槽位、NaN padding、非默认stream和只读后写回。两模式均回答4.2和56。OFF/FUSED共32个模型进程生命周期全部正常退出0，无强杀、重启或驱动修改。

局部完整Attention微测试约0.814→0.214ms；它包含Python enqueue间隙，shape与正式各层并不完全相同，不能把差值乘40当严格可消除全局占比。本项严格关键路径上限和兑现率仍未知。没有做API/MathQA/固定前缀数值门槛，未完成模型精度验收。

## 可追溯产物与后续

- `04_性能/t02_swa_*_v1_raw.tar.gz`：官方EvalScope原报告、SQLite、请求/响应、配置和角色日志。
- `03_正确性/completed_basic_raw.tar.gz`：完整算子检查与脚本。
- `05_trace/processed_trace_bundle_v1.receipt.json`：完整大trace的NFS路径/大小/SHA，以及本地代表窗口的收据。
- `06_图/BEFORE_D0`、`AFTER_D0`：实际Perfetto时间线与COUNT火焰图。UI首次加载超时已留诊断，换用已加载解析器的本机浏览器缓存后完成，不曾因此重跑推理。

后续以本项为累计最佳继续共享专家、小型投影与更大无缓存管理Graph范围。单独W4A8局部Graph微测试更慢，未晋级。CED、缓存、多请求、并行布局、DSpark等到T14的剩余项继续，不把目录编号当作任务完成度；不提MR，不宣称60%SOL。
