<!-- Copyright 2026 The xLLM Authors. All Rights Reserved. -->
# DSpark 1P1D 长时稳定性修复实验（本轮5小时验收通过）

## 最新状态（2026-10-07 14:54）

下文按实验时间保留历史判断；早期段落的“待验证”以此处及文末最新结果为准。

- 中间组合（仅prefix guard和Control Arena修复）已主动排空结束：4小时7分23秒、
  5904/5904严格通过；第四小时B1 TPOT8.865ms，B16约427tok/s。
  **这是主动提前结束的中间验证，不是5小时通过，旧组合的后测GSM未执行。**
- 全部修复已切换到最终Control/P/D候选：prefix-state guard、FFI tuple、HTTP Arena、
  Rust分词数组所有权。最终组合已通过真实1P1D混合回归44/44。
- FP8 MoE/linear的FFI tuple所有权修复已生成独立P/D候选；96单机68/68长计数、
  混合矩阵44/44通过，官方协议GSM前100题98/100。64请求存活堆增量由每rank
  128～130MiB降至0.04～1.05MiB，TPOT同输入对照8.968→8.917ms，无可见回退。
- 最终组合正式性能已通过：30K/1024、B1 TPOT8.7546ms，B16吞吐431.494tok/s，
  为B1的4.6975倍；两档各64/64严格通过，实际cached=0。
- 最终PD全量GSM已完成：**1275/1319=96.6641%**，1319条均无传输错误。
  对错及finish reason逐题与已验证基线一致，没有新增错题/截断/空输出。
  基线原有119、434两题达到1024上限，仍计为错误，不宣称它们已修复。
- 最终5小时负载及排空已完成：**7312/7312严格正确**，独立完整输出审计通过，实际cached=0，
  无传输/严格输出失败或监测告警。负载含排空共18026.601秒，没有中途重启。
  5轮B1小时探针TPOT均在8.713～8.791ms，113轮B16吞吐430.47～433.82tok/s。
  全量GSM后测已通过：**1275/1319=96.6641%**，与本次压测前逐题对错及结束原因相同；
  重复/穿插探针5/5通过，90秒空闲尾段完成。老化后正式B1/B16复测也已通过：
  **B1 TPOT8.7708ms，B16吞吐431.850tok/s，为B1的4.7011倍**，两档各64/64严格正确。
  完整证据包已封存校验11120文件。额外256/256请求严格通过，同96个arena前后对照表明，
  新增管理空间2.57MiB中93.3%已经是可复用空闲空间；存活量上界仅增加约176KiB。
  随后3次空闲采样，管理空间不再增长，存活量上界约50秒变化5.3KiB。
  已定位的FFI/HTTP/Rust所有权泄漏已修复；RSS并非零增长，不宣称全进程无限时长无任何泄漏。
  自动流程不包含老化后B1/B16重测；本次已另行完成同进程正式复测，未重启模型服务。
- 当前最终服务仍为94P/90D/92Control；96诊断已结束。全过程未resetGPU。

## 目标与基线

基于发布提交 `654e7e16fdebc39cf69b5fedb96ea8f05717479a`，独立分支
`bugfix/dspark-pd-stability-20261007`。不 reset GPU、不更改模型/评分协议，
保持同协议 30K/1024 B1 TPOT <9 ms、B16 聚合吞吐/B1 >=4.5、GSM8K >96%。
最终需要完整长时压测验证；重启恢复、少量用例通过均不能算修复完成。

原压测证据：
`/data/nfs_shared/jiaming.wang/pd_v128_30k_20261005/stability_5h_20261006_1854`。
本轮根目录：
`/data/nfs_shared/jiaming.wang/pd_v128_30k_20261005/stability_fix_20261007`。

原测试3小时24分，4880请求中4879严格通过。B1 TPOT 8.852 ms，B16输出吞吐
427.960 tok/s；发现 `stress-0378` 跳号。压测后 GSM8K 1274/1319（96.588%），
`gsm8k-1077` 从答案145退化成重复续写。P/D/Control RSS均增长，尚未证明泄漏
或与上述错误存在因果关系。

## 2026-10-07：保留故障现场的定向复测

P=94/wjm_xllm，D=90/wjm_xllm，Control=92/cmz；PID、启动时间和二进制均与
原压测一致，未重启。检查GPU空闲后启动唯一诊断客户端。
使用原始冻结payload，不修改prompt或输出上限；每次重复有独立记录，不覆盖失败。

| 实验 | 结果 | 可得结论 |
|---|---|---|
| 单独重复1077，3次 | 3次均异常续写至1024；cached=128 | 老化现场可稳定复现 |
| 1075→1076→1077 | 1075/1076正常回答，1077仍异常 | 不是所有短请求普遍失败 |
| 原stress-0378，B1重复3次 | 3/3严格通过；首次cached=0 | 跳号尚未定向复现 |
| 原368～383，B16两轮 | 32/32严格通过，含两次0378 | 不能据此宣称长时错误已修复 |
| 长请求后重复1077，3次 | 3次异常；至少前两次cached=0 | 不能仅归因于Prefix命中 |

压测前/后原1077两次请求也都cached=0；前为165 tokens正常stop，后为1024 tokens
length，文本开头已从`We can break...`变成`We need to solve...`，随后异常续写。
这里是文本首次差异，不是已定位某层/某个真实token ID的首次分叉。

证据：`aged_repro_r1/plan.json`、每次原始响应及phase summary、`complete.json`，
脚本`tools/reproduce.py`保留输入与自身hash。没有实际物理batch日志对齐前，
“B16”只表示客户端并发上限。

## 下一组控制变量

只重启P，保持P二进制/配置、D及Control不变。先保存原P的proc status/maps/smaps
和bvar，再停止准确匹配的8个P rank；不停止其他服务。随后复测同一1077，区分
问题主要随P状态还是D状态保留。这是诊断重启，不是稳定性修复方案。

主机RSS另需allocator/对象存活证据。BRPC `/memory`返回未启用tcmalloc/jemalloc，
当前仅凭RSS曲线不能区分对象滞留与glibc arena高水位。已保存只读`/vars`及proc证据。

## 控制变量结果与候选修复

| 只改变的变量 | 1077结果 | 说明 |
|---|---|---|
| P重启，D/Control保持老进程 | 3次1077仍异常续写 | P重启不能解决；配置对比一致 |
| 再重启D，原Graph/Prefix/Overlap开关不变 | 首次正确145；第二次只输出We | 已缩小为两次请求的快速复现 |
| 新D只关闭Graph | 首次提前结束；重复仍只输出We | 不能认定为Graph独有问题 |
| 新D保留Graph，只关闭D Prefix Cache | 3次1077均正确145；1075/1076也正常 | 问题指向D侧前缀状态复用；P缓存仍开启 |

对应`fresh_p_aged_d_r1`、`fresh_d_same_r1`、`d_graph_off_r1`、
`d_graph_on_noprefix_r1`。上述后两组均与新D原配置作对照，不把同时切换两个开关
误称为相邻两组单变量。P、Control及模型/算子不变，所有原始输入SHA一致。
1077重复请求的cached=128是P上报值，不代表已经测量了D的命中长度。

### Decode状态快照候选

日志证实D执行了`DSV4_CHUNK_PREFIX_SNAPSHOT before=0 end=212 prefix=128`。
`deepseek_v4_prefix_cache_policy.cpp`没有检查实例角色：
申请中间边界快照后，结束请求时仅凭已计算长度便标记ready。
但PD Decode只PULL最终prompt状态，不执行产生中间快照的Prefill forward。
复用时该快照可能覆盖刚导入的正确状态。

独立候选修改`initialize_deepseek_v4_prefix_cache()`，不为Decode创建这种自定义
快照管理器；正常Composite C4/C128缓存、P侧缓存、Graph/overlap保持不变。
尚需在**D Prefix Cache重新开启**时通过二进制回归，不能把当前配置绕过当作最终修复。
新增CPU状态管理回归，检查Decode不发布/恢复未写入快照，Prefill仍能创建快照。

该回归已分别链接原Decode `libblock.a`和候选归档执行：

- 原代码：Decode测试失败，第二个212-token请求在没有有效快照生产的情况下，
  `kv_cache_tokens_num()`被恢复为128，并产生恢复copy；Prefill测试通过。
- 候选：2/2通过。只在初始化时按`instance_is_decode()`禁用自定义快照管理器，
  不增加逐token同步或计算；性能影响仍以实际回归为准。
- 构建机28，产物`/tmp/pd_prefix_snapshot_guard_20261007/build_r2/xllm`，
  SHA `126cdaaedc3bc145bc2f45782d8af1b6ecb0233168f93889ba49b7678cfbd484`。
  只替换`deepseek_v4_prefix_cache_policy.cpp.o`，224项基线链接输入hash一致。
- 单测原始证据`/tmp/pd_prefix_snapshot_guard_20261007/tests_r1/`。
  第一次编译直接调用mcc wrapper缺少旧CMake记录之外的header路径，未生成object；
  r2恢复已有`/opt/xllm-build-deps/compiler_launcher.sh`，复用依赖完成构建。

关闭D Prefix Cache的扩展混合回归`noprefix_mixed_r1`已完成：35/35计数请求严格通过，
1077单独/短历史/30K后共7次均回答145。此结果仍是配置诊断，不代替代码候选验收。

### Control protobuf生命周期候选

BRPC默认`DefaultRpcPBMessageFactory`通过`Message::New()`创建HTTP响应，没有Arena。
旧ChatCompletions用`Arena::CreateMessage(nullptr)`创建额外请求/响应，但
`StreamCallData`析构未释放它们。独立Control候选给这两个消息显式Arena所有者，
用aliasing shared_ptr绑定异步Call生命周期；解析/调度错误路径也自动回收。
仅改Chat API当前链路，不声称其他API或P/D的RSS问题同时解决。

在92/cmz复用依赖构建：`/tmp/pd_control_arena_20261007/build_r2/`。
binary SHA `cb49fd52d68c7f7228d1894d31d3297e209de9be3d2f1608a8a70f13157cc368`；
3/3 CPU单元测试通过，依赖hash未变化，尚未替换原Control，也未做负载RSS验收。
首次构建测试链接因缓存没有gtest_main失败，r2使用明确test main和已有gtest，通过。

诊断停止服务时，部分rank能收到TERM退出，但P/D rank0超过一分钟仍未退出，
核验准确exe/role/PID后才KILL该rank；没有GPU reset、没有杀其他服务。
这是停服过程的观察，不计作原压测期间发生hang。

## 候选上线后的验证（02:56更新，仍未完成长时验收）

- D已切换到`126cdaae...`候选，PID1088407～1088414；
  Graph=true、Overlap=true、Prefix Cache=true，与原配置逐项核对一致。
- `guard_mixed_r1/independent_audit.json`：44/44通过，其中35条30K计数严格通过，
  1077共7次均回答145，1075/1076各1次正确。长请求后1077也正常。
- 3次B1计数TPOT依次为11.645、8.737、8.753 ms，第一条含新长序列Graph准备。
  后两条均为cached=19456的热请求，不能替代正式冷输入性能验收。
- Control切换到`cb49fd52...`，PID2594658，P/D未重启；启动后真实HTTP和5条推理
  均正常。1077与切换前的同payload/cached=128输出正文一致。
- P仍为`caf3db8f...`，94/wjm_xllm，PID1648226～1648233。

新5小时压力窗口从**2026-10-07 02:56:27**开始（epoch1791312987.745），
目录`soak_validation_r1/`，预计07:56左右结束发压，再追加全量GSM8K。
正式GSM8K分数尚未产生，不能引用原版本96.588%作为修复版验收成绩。

保持原1600条输入与每条30000/1024协议，B16每轮64条，每小时B1探针16条；
首错停止新提交并排空在途请求，不重试、不重启、不reset GPU。
客户端增加响应ID与完整带时间戳的SSE事件留证，**不是模型真实token-ID dump**。
按实际cached_tokens区分缓存情况，历史缓存仍在，不能直接称冷启动配对测试。
新客户端SHA记录在`frozen/client_input.sha256`与`stress/plan.json`。

P/D每分钟资源采样就在`soak_validation_r1/monitor_*`。
Control原始采样在92/cmz：
`/tmp/pd_control_arena_20261007/soak_monitor_r1/monitor_CONTROL/`，
共享目录`monitor_CONTROL/`只保存启动确认与初始身份快照，结束后需要归档完整样本。
Control不挂载jiaming.wang目录，因此其本地告警需单独巡检；HTTP故障也会使客户端停压。
压力结束后GSM8K生成设置额外3600秒总超时，CPU官方评分仍需单独执行。

待完成：5小时严格正确性、B1/B16吞吐和TPOT、Control内存趋势、P/D RSS增长定位、
全量GSM8K。原来的偶发30K跳号尚未单独定位到具体数值/状态错误点，不能提前宣布解决。

### 新压测首轮B1（非5小时最终结果）

`soak_validation_r1/stress/rounds.jsonl` round1：16/16严格通过，所有cached=0；
180.7616秒，实际16384输出tokens，聚合90.6387 tok/s。
TTFT均值2228.947 ms，TPOT均值8.849840 ms、P95 9.097058 ms。
均值与原压测8.852 ms基本一致，没有将热命中或Graph首次准备作为稳态成绩。

### 新压测首轮B16（非5小时最终结果）

round2：64/64严格通过，所有cached=0；153.1735秒、65536输出tokens，
聚合**427.8547 tok/s**；TTFT均值15105.308 ms，TPOT均值19.906434 ms。
相对本轮B1探针的90.6387 tok/s为**4.72044倍**。这是原压测协议中的不同输入分段，
不是重新完成了“同一64条输入、每臂新P进程”的成对验收。
初步未见吞吐/TPOT回退；后续轮次、5小时正确性与资源上界仍待检查。
Control在最初几分钟从约380 MiB增长到约1.7 GiB，存在并发预热/allocator高水位，
不能仅据单测通过就称现场RSS问题已解决。

## 03:29：保持压力运行的只读内存定位

本轮延续时重新核验压力driver1097616、Python子进程1097618及资源采集器存活，
没有因旧状态文件或观察超时重新启动负载。已通过726条，暂无错误；最近完整B16轮次
吞吐约425～430 tok/s。仍不足以替代5小时结果。

### 内存增长落在哪一类映射

读取`/proc/PID/smaps`，不attach、不暂停进程、不注入代码。两次约157秒间隔的快照：

| 进程 | RSS增量 | 增长类别 |
|---|---:|---|
| D rank0 | 32128 KiB | 全部匿名映射 |
| D rank1 | 13120 KiB | 全部匿名映射 |
| P rank0 | 20484 KiB | 全部匿名映射 |
| P rank1 | 19996 KiB | 全部匿名映射 |

库文件、`/dev/dri`映射及主`[heap]`在该窗口没有增长，多数增量集中在单个匿名区域。
这把排查范围缩小到CPU分配/持有，**不能单凭64 MiB对齐就断言是glibc泄漏**。
证据在`soak_validation_r1/maps_{P,D}_r{1,2}`与`maps_*_r1_r2_diff.json`。

另外进行了有读次数/1秒预算的allocator元数据只读探针，没有读取/导出用户文本，
没有ptrace或写目标内存。版本布局参考[glibc 2.35 arena源码](https://github.com/bminor/glibc/blob/glibc-2.35/malloc/arena.c)
和[malloc源码](https://github.com/bminor/glibc/blob/glibc-2.35/malloc/malloc.c)。
由于实时free-list会并发变化，探针遇到越界/尺寸变化或预算限制，所有结果均标记为
`inconclusive`，**不作为精确存活字节数或泄漏的证据**；没有进一步锁定/暂停服务来取数。

### CPU FFI包装生命周期排除测试

新增`tests/core/kernels/musa/musa_tvmffi_lifetime_test.cpp`，在28构建机以CPU tensor测试：

- owning包装在最后一个FFI引用释放后释放Torch storage；
- borrowed包装只持有metadata，不持有Torch storage；
- 10000次非连续tensor包装后引用数回到初始值；
- 预热后100000次owning/borrowed包装的活跃堆增长不超过1 MiB。

**4/4通过**。只排除基础包装函数的直接累积，不覆盖实际Mate算子内部、Graph分配记录、
异步事件和驱动持有，不能据此排除整个FFI链路。
使用冻结Decode链接依赖；90实际映射的`libtvm_ffi.so`与该测试一致，SHA均为
`9bc5641eb143a46216df03351b61f9d2075ee7037845535984d35180be61cd68`。
测试binary SHA `a9b0a29b098ea0a772df72b0e46787a4baf8d4143a26b50c46a71999b454714b`。
证据包`ffi_lifetime_evidence_20261007.tar.gz`已归档到本实验根。

Control并发预热后约2.35 GiB，03:05～03:24采样增长仅数MiB，较初期趋于平稳，
但仍需等完整压力窗口确认。96的GPU空闲但根盘Avail=0；92的GPU被vLLM占用，
没有停止该任务，也没有在96启动额外GPU实验或擅自清盘。

## 04:00～05:40：进一步定位CPU内存增长

### 主压测保持不变

一小时B1检查16/16通过：TTFT 2184.487 ms，TPOT 8.793580 ms，全部cached=0。
两小时B1检查16/16通过：TTFT 2174.914 ms，TPOT 8.830976 ms，全部cached=0。
截至约05:37，3861条请求严格通过，近期完整B16轮次约425～430 tok/s。
主P/D/Control未为下列诊断重启，也未加入profiler。

第10～60分钟：Control rank0 RSS由2400.797增至2412.785 MiB（约+12 MiB），
比此前持续增长明显缓和，但完整长测尚未结束。P rank0由8926.457增至9295.875 MiB，
D rank1由8741.363增至8978.773 MiB，P/D的增长仍需解决。
证据：`soak_validation_r1/resource_{PREFILL,DECODE}_hour1.json`；
Control对应文件在92/cmz的`/tmp/pd_control_arena_20261007/soak_monitor_r1/`。

### 96基础微测：不能把pinned统计虚增当作真实泄漏

Docker可执行，Torch/MUSA库hash与90相同；临时文件用`/dev/shm`，数据/缓存/日志用共享盘，
未清理根盘。只用空闲0号卡，四个独立进程各执行20000次，结束后GPU无进程：

| 操作 | RSS增量 |
|---|---:|
| pinned分配/释放 | 12 KiB |
| pinned H2D | 0 KiB |
| 普通matmul（固定out） | 152 KiB |
| Graph matmul replay | 208 KiB |

这不能覆盖完整模型的临时Tensor和实际Mate调用。H2D的pinned allocated统计虚增约5.2GB，
reserved只增加2.25MiB且RSS不变。检查当前Torch头文件发现：
`process_events()`调用`process_events_for_specific_size(-1)`，释放时却执行
`allocated_bytes_bucket_stats[index].decrease(size)`，用了-1而不是`block->size_`。
因此此计数不可靠，不用它判定P/D内存泄漏；**没有修改生产依赖库**。
证据：`allocator_micro_96_r1/`，包括输入、采样、各阶段结果和`evidence.sha256`。

### 完整单机对照：每请求约2MiB存活堆增长

为避免大规模NFS读取影响PD压测，先用`fincore`确认96的48个模型分片全部已驻留页缓存。
随后使用相同`126cdaae...`binary/AOT，在96的39800端口启动独立单机TP8 DSpark，
**PD=false，不注册到原Control**。N6、Graph/Overlap/Prefix开启，chunk=8192；
它是内存对照，不是1P1D性能验收。额外的堆观察库每10秒读统计，不清缓存、不做GPU同步。

`single_heap_96_r1/independent_heap_audit.json`：4条预热+64条正式请求，**68/68严格通过**。
比较预热后与正式请求后各30秒空闲窗口：各rank的`malloc_used`仍增加约
**128.18～129.56 MiB，约2.0 MiB/请求**，`malloc_mmap`、pinned reserved不变。
所以不能仅解释为“空闲内存留在allocator里”，且该现象在不启用PD时也能复现。
尚需区分具体长期持有对象与缓存，不将该数据单独当成所有路径都已确定的泄漏根因。

### 分配栈指向FP8 MoE的FFI参数封装

仅在96诊断进程rank1使用独立解包的jemalloc 5.2.1，其他rank保留原allocator；
包来自配置中的Ubuntu仓库，hash见`heap_profiler_deps/artifacts.sha256`，未执行系统安装/升级。
采样依据[jemalloc profiling接口](https://jemalloc.net/jemalloc.3.html)，关闭`prof_accum`，
不把采样器引入的开销作为正式性能结果。

- 启动后再开启采样的`single_heap_96_je_r1`：36/36正确，但profile为空，结果不具定位效力。
  同库独立自测能记录16个2MiB分配，所以改用从进程启动即采样的独立实例。
- `single_heap_96_je_startup_r1`抓到有效分配栈。240→300秒差分约7.86MB，
  其中约86.7%沿`DeepseekV4MoEImpl::forward_grouped_fp8 → ragged_moe_gemm_fp8`，
  约80%经过`to_ffi_borrowed_tensor`及其下游包装；另有FP8 linear的tuple封装。
- `analysis_early/top.txt`、`borrowed_cum.txt`保留原始分析。工具对部分DWARF格式发出警告，
  不能仅凭个别错误符号名定责；后续以最小CPU重现进一步确认。

当前FFI的`std::tuple`转换路径在`CopyToAnyView`中创建并转移了一个拥有引用的Tuple容器，
而接收方是非拥有的`AnyView`，缺少回收所有者。这解释了为什么只测试
`to_ffi_borrowed_tensor`自身时4/4通过，而实际调用后仍累积容器和Tensor元数据。

### 最小重现与修复候选

扩展`musa_tvmffi_lifetime_test.cpp`，真正跨一次`ffi::Function`调用边界：

- 旧`std::tuple`参数：调用后元素引用数为2而非1，测试失败；整数tuple重复调用堆增长，
  另一个测试也失败。原来的四项包装基础测试仍通过。
- 显式`ffi::Tuple`参数：**6/6通过**，引用数恢复，100000次形状参数调用未超过1MiB增长门限。

修复仅调整参数容器的所有权，不改Tensor数据、量化粒度、算子选择或GPU数学运算：

1. `musa_moe_gemm.cpp`：masked / contiguous / ragged FP8三条路径；
2. `fp8_block_gemm.cpp`：量化粒度tuple；
3. `deepseek_v4_masked_fp8_moe_gemm.cpp`：masked备用路径。

新候选同时保留前面的Decode prefix-state guard，每个角色重编4个translation unit，
其余224项链接依赖hash一致，AOT/model不变。产物在28：
`/tmp/pd_ffi_tuple_fix_20261007/`，CPU A/B证据在`tuple_tests/`。

| 角色 | 完整debug binary SHA256 | 运行副本SHA256 |
|---|---|---|
| D | `8cf17e56232769594ac43e3ea0d2483e4937041258c99bd521a18b7003a63be4` | `634242217369665985af16c08c215d51295d32a2211a3b3099665125e7a98f74` |
| P | `1ad760005129a60a6d56bbfd4d7170d7abb558e398c96f58b186f9fb3798bd7f` | `1f4b027be1cca75bfdce77c56dddd0e91ae676c48fdd1238bf077d4dd1b342e1` |

运行副本仅`strip --strip-debug`，逐一验证所有SHF_ALLOC section的地址、尺寸、内容hash
完全一致，完整debug binary保留；细节见`runtime_manifest.json`。
**这些新FFI候选尚未完成GPU模型回归、PD性能或新一轮长时验收，不能提前称RSS问题已解决。**

## 05:52：新FFI候选进入真实模型回归

包已中转至共享盘`ffi_tuple_candidates/`并核验P/D运行副本SHA。
96新实验目录`single_heap_96_fixed_r1/`，实际Decode来源运行副本为`63424221...`，
单机PD=false、Graph/Overlap/Prefix均开启，**恢复原glibc，不预加载jemalloc**，
仅保留与旧单机对照相同的低频堆观察库。
rank PID377792、377794、377796、377798、377800、377802、377804、377806，
监督进程377613，串行客户端381896。使用与旧单机对照相同的4条预热、64条正式输入。

首批真实推理已通过，完整64条及空闲窗口差分尚未完成，不能提前宣布内存曲线已修好。
首次启动尝试因包尚未传完而在binary hash预检处退出，没有创建server或启动GPU服务；
保留`launch.log`，传输完成后另记`launch_attempt2.log`，没有用旧binary冒充新候选。

主1P1D压力实例仍是P=`caf3db8f...`、D=`126cdaae...`、Control=`cb49fd52...`，
截至约05:57已连续3小时、4337条严格通过。**它不包含新的FFI tuple修复，
不能把这轮结果直接计作`63424221/1f4b027b`组合的长时验收。**
完成96回归后，还需对最终P/D组合重新验证性能、全量GSM8K与长时稳定性。

## 06:15：FFI修复后的同输入内存对照完成

`single_heap_96_fixed_r1/independent_heap_audit.json`已独立校验完整输出：
4条预热+64条正式请求，**68/68严格通过**。每条均为30000输入/1024输出、
512个连续整数逐项正确，finish_reason=length，没有重试或省略失败样本。

| 同一64条正式请求，前后空闲窗口比较 | 旧126c候选 | 新6342候选 |
|---|---:|---:|
| 各rank存活堆`malloc_used`增量 | 128.18～129.56 MiB | 0.044～1.045 MiB |
| 非driver rank1增量 | 128.52 MiB | 0.044 MiB |
| `malloc_mmap`增量 | 0 | 0 |

新候选仍有少量缓存/管理开销，不能据此声称无限运行完全无增长；但旧版本约2MiB/请求的
线性增长已在这组同输入、同allocator、同观察方式的真实模型实验中消失。
这与FFI tuple引用计数CPU A/B结果一致，且未关闭Graph/Overlap或替换GPU算子。

随后在同一96进程上启动冻结的GSM8K前100题，并追加
1077、1077、1075、1076、1077重复历史诊断；生成过程不读取答案。
前100题将用原SGL官方CPU评分器独立评分，重复题另列，不能冒充1319题全量成绩。

主PD仍为旧FFI组合，截至06:15约3小时19分、4732条完成且全部严格通过，
最新完整B16轮427.03 tok/s。新FFI最终P/D组合的性能、全量GSM8K、长时稳定性仍待验收。

## 06:23：新FFI候选的性能对照、GSM100及混合并发回归

同一单机96、相同4条预热/64条正式输入、相同glibc与低频heap observer，
Graph/Overlap/Prefix均开启。独立脚本确认旧/新有序请求payload SHA完全相同，
正式请求均cached_tokens=0：

| 96单机诊断（不是PD生产验收） | 旧126c | 新6342 |
|---|---:|---:|
| 正式TTFT均值 | 3583.074 ms | 3581.866 ms |
| 正式TPOT均值 | 8.9682 ms | 8.9167 ms |
| 实际输出/完整正式时段 | 80.215 tok/s | 80.560 tok/s |

该对照未观察到FFI所有权修复带来的性能回退，TPOT变化约-0.57%；小幅差异不宣称额外加速。
证据：`single_heap_old_new_comparison.json`。这不是优化Prefill版本或最终PD组合的性能成绩。

GSM8K前100题已完成，转到28/wjm_sg_dspark用未改动的SGL MathEvaluator做CPU评分：
**98/100=98%**，100条均finish=stop，0传输错误、0截断；错题37、87。
辅助numeric-boxed得97分，与官方符号评分在14题有差异，不混用两个口径。
这是100题诊断，**不能代替1319题全量验收，也不是新跑了一轮SGLang推理**。
随后5条重复探针均正常，1077多次返回145，包括cached_tokens=128的情况。

- 生成：`single_heap_96_fixed_r1/gsm100_r1/`；
- 官方评分：同目录上一级的`gsm100_r1_official_scoring/`；
- 评分包SHA：`9c8faa968db56fada3658b6adb97cbd4ff78a6f36aba9c5aababb9114733de7b`；
- 输入评分行SHA：`476d881472900953c232368e22fa1a39470534febf97a6b44db42f1e3b272521`。

同一服务不重启，继续执行35条计数+9条GSM混合诊断：**44/44通过**。
包括旧失败case378串行3次、368～383的两轮16并发，以及长计数后再次请求1077。
全部输出独立逐项验证，原始结果与校验在`single_heap_96_fixed_r1/mixed_r1/`。
计数部分记录真实cache命中，不假设这些重复请求全为冷缓存；该矩阵只用于正确性回归。

### 主PD压力实例跨过旧失败点，仍未结束

对原始SSE、完整正文、冻结payload和提交记录做独立快照校验：
`soak_validation_r1/audit_partial_after4875.json`中**4925/4925严格通过**，
均cached_tokens=0；其中case378的第75、1675、3275、4875次请求全部正确，正文SHA相同。
本次没有出现旧版第4875次的790→890跳号。

最近一次小时B1探针TPOT8.731ms、92.017tok/s；B16仍约427tok/s。
但偶发跳号的独立根因尚未因这一轮通过而被证明，仍继续完整5小时，不提前结束。
当前主PD仍没有新FFI修复，D rank1在10→180分钟RSS由8741.36增长至9555.65MiB，
故不能将其认定为最终稳定组合。后续只在本轮测试与后测客户端结束后切换P/D。

### 最终组合的下一阶段验收

已准备`stability_fix_20261007/tools/start_role_ffi.sh`并通过bash语法校验，
显式锁定P=`1f4b027b...`、D=`63424221...`及冻结launcher哈希，
清除继承的DIAG/allocator实验变量，保留既有高性能路由参数；尚未启动替换主PD。

1. 保存本轮5小时完整输出和资源记录，完成原组合的后测GSM，停止并确认客户端退出。
2. 按精确exe/role/PID停止本项目P/D及Control；按Control→Decode→Prefill启动最终候选。
3. 执行混合正确性矩阵、相同64条30K/1024的B1/B16性能验收、全量1319题GSM。
4. 以上通过后，对最终binary/config重新执行5小时压测及后测GSM，审计内存曲线。

完成条件仍为B1 TPOT<9ms、B16实际输出吞吐/B1≥4.5、GSM8K>96%，
且完整压测无严格输出错误、hang、OOM或持续异常增长。不同版本不拼接为一次验收。

96诊断在06:25结束并核实8卡无进程。356个输入/日志/结果文件已封存，
`single_heap_96_fixed_r1/sealed_evidence.json` SHA为
`8124cb3f69eb456bb83adeb58bc64ebfd4a7b848674a48d381abf10f4ccaf5dd`。
完整诊断包`ffi_fixed_single96_validation_20261007.tar.gz` SHA为
`5e0c480e42b0b7a0d4ef7d1d5d012938ed25c849d75d1d49a58f18854fe0cfd3`。
封存后不再向该96实验目录追加测试；后续使用独立新目录。

## 06:40：Control还有一处Rust分词结果数组泄漏，已完成候选修复

这与之前HTTP protobuf Arena所有权、GPU侧FFI tuple是三条不同的内存路径，不能互相替代。
只读检查正在运行的Control启动日志，明确为`Create fast tokenizer`；
`FastTokenizer::encode()`调用Rust的`tokenizers_encode()`，拷贝ID后返回，未释放源数组。
Rust实现通过`Box::into_raw(encoded.into_boxed_slice())`转移所有权，已有匹配释放函数
`tokenizers_free_encode_results()`，但C++头文件未声明、调用方未使用。
旧Control二进制反汇编也证实encode只调用分词/拷贝，没有该释放调用。

当前实际Rust库：
`/workspace/xllm-service/build-cmake331-native/xllm_service/tokenizer/tokenizers/x86_64-unknown-linux-gnu/release/librust_tokenizers.so`，
SHA `76ed63df241ceb1b109a57eef118766e5a0bc523014740c4a8337fa003f456be`；
该库已导出释放函数，无须升级、重新编译或替换Rust依赖。

### 独立修复和CPU对照

本机独立源副本：`.codex-pd-n6q7-20260927/recheck/stability_fix_20261007/control_tokenizer_candidate/`。
92/cmz独立构建：`/tmp/pd_control_tokenizer_20261007/build_r1/`，未写原workspace的build cache。

- `xllm_service/tokenizer/tokenizers/tokenizers.h`补充已有释放API的声明；
- `xllm_service/tokenizer/fast_tokenizer.cpp`用局部RAII在拷贝后释放Rust数组，异常退出也释放；
- 继续保留`ChatCompletions`的Arena修复，只增量重编2个Control translation unit；
- 增加两个`cc_test`，固定mock释放计数以及真实模型tokenizer的CPU回归。

| CPU回归 | 旧生产object | 新候选object |
|---|---:|---:|
| 释放计数/空输入/重复调用/跨调用输出保持 | 0/4通过 | 4/4通过 |
| 真实模型：相同30K分词连续256次存活堆增量 | 30,727,984 bytes | 1,856 bytes |
| 真实模型长测及空输入/中文/特殊token对齐 | 1/2通过 | 2/2通过 |
| 30K token-ID FNV1a64指纹 | 15013964437387898805 | 15013964437387898805 |

新旧输出token IDs逐项相同；FNV只作额外输出指纹，不替代产物SHA256。
对应`legacy_lifetime/fixed_lifetime/legacy_real/fixed_real.{log,json}`均保留。
新候选6/6通过，没有跳过真实模型测试。旧版本的失败是预期反例，不从验收中删掉。
这里只证明该分词数组的泄漏被修复，**尚未证明线上Control所有RSS增长均由它引起或已消失**。

最终Control候选：
`/tmp/pd_control_tokenizer_20261007/build_r1/xllm_master_serving`，
SHA `244fd53e85a3a9e5186a2956f5d2392229cd13186bf8a41170335909068298a1`。
依赖库hash前后保持一致，完整源码/命令/链接依赖记录于`build_r1/manifest.json`。
打包到共享盘：`pd_control_tokenizer_evidence_20261007.tar.gz`，
SHA `01ef15366fcce54196be80bc34a2c2223c607b51030305f337094cee47529449`。
92宿主机普通用户不能写此NFS目录，因此先从cmz拷到宿主机/tmp，再通过自己的
wjm_xllm容器写入共享盘；没有启动GPU进程、修改权限或覆盖原产物。

已准备92/cmz启动器`/tmp/pd_control_tokenizer_20261007/start_control_final.sh`，
检查binary/Rust哈希、空闲端口和etcd健康，并显式清除`XLLM_SERVICE_ASSUME_PRELINKED_PD`。
它不主动停止任何进程；当前5小时压测与后测结束后，才替换Control并重启最终P/D组合。

## 07:00：补齐最终长测的监测闭环（未更换当前压力实例）

现有中间版本压力测试的P/D监测在共享目录，Control监测在cmz本地，原客户端不能直接读取
Control的告警文件。目前仍逐次核查Control进程/监测PID；不把这一限制隐瞒成自动告警已覆盖。
最终工具独立放在`final_harness_tools/`，没有修改正在运行的`soak_validation_r1/tools/`。

- Control监测通过92宿主机Docker标准输入输出转发到自身wjm_xllm容器，再写入同一个
  共享实验目录；无需改容器挂载、权限或etcd，不涉及GPU操作。
- 三角色都记录binary SHA、PID、start_ticks与每分钟心跳。心跳超过180秒、角色进程缺失/
  换代、错误日志或监测提前退出，均阻止继续验收；压力阶段停止提交并排空已发请求。
- 串行后测若监控失效，只终止该工具自己创建的评测子进程，保留不完整结果，不能计作全量。
- 成功的5小时负载之后，独立审计原始SSE/完整正文/输入，执行1319题后测，再执行
  1077→1077→1075→1076→1077复测与离线校验，最后保留90秒空闲尾段。
  监测上限7小时用于覆盖请求排空和后测，**压力提交窗口仍是5小时**。

验证证据：

1. `final_harness_test_r3.log`：20/20 CPU故障注入测试通过，包括过期心跳、PID复用、
   转发断开、错误日志、停止/排空策略，以及“We”、截断、错误末尾答案、篡改输入拒绝测试。
2. `harness_transport_probe_r1/`：使用仍在线的旧Control真实进程快照验证跨容器传输，
   明确标注`transport_probe_only`；不是新Control上线或长测成绩。
3. `final_harness_gsm_dryrun/`：1319条冻结输入完整检查通过，**HTTP请求数为0**。
4. `harness_repeat_check_r1/`：在历史`post_control_guard_probe`的独立副本上验证离线检查器，
   5/5通过；原始响应未修改，不算新跑的GSM8K结果。

脚本逐文件hash见`final_harness_r3.sha256`，封存包`final_harness_20261007.tar.gz`的SHA：
`c6925ff9837a931ba773abf91ae14b46402215955e06e2bdbd0ae59aa5308392`。
原有冻结stream请求实现与GSM生成/评分适配器均未改变；新增的是安全监测与审计编排。

## 07:25：结束中间轮并切换最终Control/P/D，真实混合回归通过

### 为什么调整原定等待计划

全部修复及验收工具在中间轮运行超过4小时后均已就绪；中间轮仍不包含FFI和分词数组修复，
不能用于验收这些内存修复。因此主动写入`soak_validation_r1/abort_requested`，停止新提交、
排空已有请求，提前推进最终组合，而不是把剩余时间当成已完成。

中间轮实际`14843.462s`，提交/完成5904，严格通过5904，失败0。
`stress/complete.json`保持`completed_duration=false`、`soak_pass=false`，原因只有
`operator requested stop`。`orchestration_complete.json`记录stress_exit=2、gsm_generation_exit=125。
独立原始输出审计`independent_complete_output_audit.json`确认5904条全部cached=0、
正文逐项正确、SSE与归档一致，无遗漏提交；其中第4875次也正确。

第四小时B1探针16/16，TTFT2189.050ms、TPOT8.865ms、90.861tok/s。
最近完整B16轮427.110tok/s。旧代码240分钟RSS仍为P rank0 10582.652MiB、
D rank1 9846.285MiB、Control2592.586MiB，不能以精度通过宣称旧代码内存已修好。

### 受控停服和重新启动

确认客户端PID1097616/1097618均消失后，使用`stop_intermediate_owned.py`逐一验证
旧binary SHA、exe、PID、start_ticks及角色参数，再通过pidfd发送信号，避免PID复用误杀。
先TERM；P/D各自剩余的rank0约85秒后仍未退出，再次验证后只对这两个rank0发KILL。
这属于停服回收问题，不是压力期间出现的推理hang，也未宣称停服退出问题已修复。
随后分别保存90/94的GPU空闲证明，未resetGPU。

旧Control监测和停止记录封存为`pd_control_intermediate_end_20261007.tar.gz`，SHA
`795e9d2ea4cc8be9c052cfff94a5d139a7d2f5ae811047e396a9b0f6f3852da6`。
第一次新Control启动仅在端口预检查报Address already in use，进程已退出且未建立runtime目录；
保留`runtime_final_r1.launcher.log`。确认旧监听/进程消失后，用原样脚本和新日志目录r2启动成功，
没有因观察超时而重复启动服务。

| 最终角色 | 机器/容器 | 当前PID | binary SHA前缀 |
|---|---|---|---|
| Control | 92/cmz | 2595296 | 244fd53e |
| Decode | 90/wjm_xllm | 1139014～1139021 | 63424221 |
| Prefill | 94/wjm_xllm | 1696438～1696445 | 1f4b027b |

按Control RPC→Decode健康→Prefill顺序启动。Control在两角色注册后才开放HTTP；
已验证两角色`/health`、Control`/v1/models`，并通过实际推理，未开启ASSUME_PRELINKED绕过。
角色目录：`final_ffi_pd_r1/{PREFILL,DECODE}`；Control运行目录：
`/tmp/pd_control_tokenizer_20261007/runtime_final_r2`。

### 最终组合的真实回归和在途验收

`final_ffi_pd_r1/acceptance/mixed_r1/independent_audit.json`：**44/44通过**，
含35个30K/1024完整计数、两轮16并发，以及9个GSM历史/重复探针；1077恢复145且正常stop。
这是最终三角色组合的真实推理，不是CPU模拟；仍不是1319题全量成绩。

新监测全部实时汇总到`final_ffi_pd_r1/acceptance/monitor_{PREFILL,DECODE,CONTROL}`，
三角色心跳检查通过。Decode monitor PID1147217，Prefill monitor PID1703773，
92宿主机Control桥接PID3187383。

正式预验收由PID1148992运行`preacceptance_tools/run_preacceptance.py`：
每个并发档16条预热、64条正式，同一320～383输入、严格要求实际cached=0；
随后生成全量1319题供原SGL评分器评分。通用客户端summary中的旧“8GPU aggregate”文字未改，
独立审计明确按当前P8+D8共16GPU解释，吞吐始终按实际输出token/完整客户端时段计算。
结果分别保存在`logs/final_ffi_pd_r1_b{1,16}_{warmup,formal}`和
`final_ffi_pd_r1/acceptance/gsm_before`，后者尚在等待性能阶段完成。

当前预热阶段开始返回正确输出；正式性能、全量准确率、最终5小时稳定性仍未验收完成。
本次计划调整不改变目标，也不把不同binary/config的时长或成绩合并。

## 07:52：最终组合正式性能门槛通过，全量GSM进行中

两档分别16条预热、64条正式；正式输入均为320～383同一组冻结case，输入30000、输出1024，
全部512个整数逐项检查，实际cached_tokens=0。结果来自完整summary，不采用最后一条progress的
略短计时时段。独立再审计`acceptance/independent_pair_audit.json`确认输入集合及payload完全相同。

| 最终1P1D，30K/1024冷缓存 | B1 | B16（客户端并发上限16） |
|---|---:|---:|
| 正式严格通过 | 64/64 | 64/64 |
| 实际输出token总数 | 65536 | 65536 |
| 完整客户端时段 | 713.463s | 151.881s |
| 聚合输出吞吐 | 91.856tok/s | 431.494tok/s |
| 平均TTFT | 2183.681ms | 14804.504ms |
| 平均TPOT | 8.7546ms | 19.8648ms |

吞吐倍率为**4.6975倍**，按实际输出token/完整公共时段计算，不用`batch/TPOT`推算。
达到当前目标B1 TPOT<9ms、B16吞吐≥4.5倍；该目标并不等价于B16单请求TPOT<9ms。
B1/B16输入payload对齐SHA：`ad076bca62687afe22844e0ae784045c90f042c314a4aa2bd4b5638ee8886be6`。
注意此处是30K冷缓存，不能与2.5K热缓存约220ms TTFT混为一谈。

### 当前准确率验证

1319题生成客户端PID1152067已启动，预验收监督PID1148992保持运行。
从正在进行的同一轮全量生成中，只读提取前100个已完成结果，核对原始HTTP body/hash和冻结输入，
再转28/wjm_sg_dspark用原SGL MathEvaluator评分：**98/100**，错误37、87；
100条均stop，0传输错误、0截断，与此前96单机候选的前100题错题集合一致。
没有重发模型请求、改输入或过滤错误；该快照始终标记partial_diagnostic。
后续仍以1319题全量评分为准，不以100题结果启动“已达全量准确率目标”的结论。

输入评分文件SHA：`984038ece5d7aec7d43a59c2fdd6806757946881c8064229afff16b2f60d1db6`；
评分证据包SHA：`c20eeed807f0d95f4136bfd5496f731be21aee86d45f8ba0a6499d1e468b8d39`，
保存在`acceptance/gsm_prefix100_r1/`。完整GSM仍在`acceptance/gsm_before/`生成，
最终5小时负载尚未启动。

### 代码、构建与实际进程对应关系

`final_ffi_pd_r1/provenance/verified_provenance.json`通过以下检查：

- Windows独立worktree四个改动文件、28完整源码快照、P/D各自4个重编translation unit的SHA一致；
- 两份runtime binary与debug binary的strip映射记录、当前8+8+1进程监测身份一致；
- 当前Graph/Prefix、N6、P/D overlap差异、batch/chunk、Mooncake PULL和模型路径均符合冻结配置；
- 16个GPU worker只映射同一个预期`libtvm_ffi.so`路径，未预加载heap observer或jemalloc。

进程配置采集只读取/proc元数据，不调用GPU或读取进程数据内容。归档核对在空闲96上执行，
没有重新编译、重启服务或改变评测参数。完整源码包`pd_ffi_tuple_source_20261007.tar.gz`已放到共享盘，
SHA：`4e80a0f919dd01c9f811dca6ca2c7a87a930e1253127cda9aa4a7904abe1192c`。
仍明确属于复用冻结链接依赖的增量构建，不声称已经做过无缓存clean build；
共享库本次核对为映射路径/stat身份，不将其扩大解释为重新计算了全部库内容hash。

## 08:20：全量准确率门槛通过，最终5小时稳定性验证已启动

### 全量GSM结果及原有截断问题的核实

最终组合按冻结1319题、temperature=0、top_p=0.95、max_tokens=1024、thinking=false生成完成，
传输错误0。仍使用原SGL MathEvaluator：**1275/1319=96.6641395%**。
辅助numeric-boxed为1254/1319=95.0720%，不替换预先固定的符号评分口径；
这21个评分差异与历史基线一致，不通过事后换评分方式凑门槛。

发现2题finish=length后，没有忽略它们，而是与原压测前已验证的`pd_guard_full`完整结果逐题比较：

- 119和434在基线中也达到1024上限；本次没有新增截断，且仍计为错误；
- 全部1319题的symbolic_correct及finish reason逐题相同，空输出均为0；
- 输入payload和参考答案逐题相同，1297题生成正文完全一致；其余22题正文有差异，
  包括两道截断题，因此不宣称新旧结果逐字全相同；
- 1077本次正常输出145，164tokens、stop，未重现压测后的异常续写。

两道题涉及模型对题意反复推理，本轮没有修复这两个原有模型局限。
此前启动门禁额外设置的“所有题零截断”并非已验证历史基线的条件，现依据完整证据校正为：
全量准确率>96%、无新增截断/空输出、无原本正确题变错、无stop退化，所有1319题仍在分母内。
这既不删题、改prompt、放宽输出上限，也不把已有错误改判为正确。
门禁8项CPU测试通过，特意检查即使总分仍>96%，1077等新增退化也不能解锁长测。

证据在`acceptance/gsm_before_official/logs/pd_final_gsm1319_before_20261007/`：
`*.summary.json`、`*.scored.jsonl`、`baseline_comparison.json`。
当前评分输入SHA：`eefc3900243c1195c1807e0238ee43d8d8ab7fcdeb388102d99a068a44fc8300`；
当前scored JSONL SHA：`0f83037b914870c1bd8ac3bbe8ee22e5dd08d90d2bd38ab3b0486ee1b0e535c1`；
冻结基线scored SHA：`11a9513bc4db95688b0a66d0bd6d9d95ff1608a167d7c8e35327122f3e145c92`；
完整评分证据包SHA：`146feee2b85bcf7ed94f58928bf614a42114af1e77206b5a835cf7fe82b5d7e1`。

### 最终长测：同一批服务进程，独立新的监测窗口

预验收客户端1148992/1152067已正常结束。只结束其资源监测，不重启任何模型服务，
为最终负载启用新监测窗口，避免监测器7小时时限被预验收消耗。

- 根目录：`final_ffi_pd_r1/soak_final_r1`；
- 起始epoch：1791332134.838139，即2026-10-07 08:15:34北京时间；
- 计划提交截止epoch：1791350134.838139，即13:15:34，之后排空未完成请求；
- 监督PID1159404，压力客户端PID1159406（90/wjm_xllm）；
- Decode monitor PID1156107，Prefill monitor PID1711563；
- Control monitor为92/cmz PID2595647，宿主机桥接PID3193218；
- Control/P/D仍为2595296、1139014～1139021、1696438～1696445，start_ticks亦相同。

`acceptance_gate.json`绑定正式性能审计、全量评分、基线逐题比较及17个服务进程身份。
启动前逐项核验冻结harness SHA，未更换模型、算子、binary或Graph/Overlap/Prefix参数。
计划仍循环同一1600个30K/1024输入，B16负载并每小时插入16条B1探针，首次异常停发并排空。
成功完成5小时后自动全量1319题后测、1077重复回归、90秒空闲尾段；之后仍需独立评分和资源审计。

首轮B1探针16/16正确：TTFT2171.432ms、TPOT8.7907ms、91.624tok/s，实际cached=0。
截至08:20已完成43条且全部严格通过，三角色告警为空。
这只是起步检查，不是5小时或最终内存稳定性通过；必要的压测后同协议性能复核也不得用预验收代替。

## 08:37：长测继续，准备同进程压测后性能复核

最终压力客户端1159406及监督1159404均持续存活；约21分钟时473/473严格正确，告警为空。
最新完整B16轮64/64正确、432.333tok/s、TPOT19.905ms，实际cached=0。
没有改参数、重新启动模型或resetGPU。

`soak_final_r1/resource_{PREFILL,DECODE,CONTROL}_10min.json`已保存早期快照。
以下只是该窗口观察值，**不是稳定性门槛或泄漏已消失的结论**：

| 进程 | 负载开始附近RSS | 10分钟附近RSS |
|---|---:|---:|
| P rank0 | 8862.750MiB | 8871.344MiB |
| D rank1 | 8693.102MiB | 8697.988MiB |
| Control | 2334.594MiB | 2338.664MiB |

部分其他rank有更大的初期增量，例如D rank0约67MiB、部分P rank约43～52MiB，
尚不能仅据10分钟窗口区分allocator/cache高水位与持续增长。必须结合后续小时级曲线，
不能挑选低增长rank代表全部进程。

新增但**尚未执行**的`preacceptance_tools/run_post_performance.py`用于最终老化后复核：

1. 要求5小时严格负载完成、独立全文审计通过、全量后测>96%且无新增逐题退化、重复题5/5通过；
2. 使用独立新结果目录与新监测窗口，但必须保持Control/P/D全部PID、exe、start_ticks和binary不变；
3. 再跑相同304～319预热、320～383正式64题，B1/B16均要求实际cached=0，验证完整输出及相同吞吐门槛。

脚本SHA：`9b14364b8ae30d844f0803e92f29d37b626f107df7aff243430294c908c67254`。
当前只检查语法/命令入口，并做了“长测尚未完成时必须拒绝”的CPU负例，
`post_perf_early_guard_check.log`记录预期拒绝、未创建性能计划，也未发送任何推理请求。
这不冒充压测后性能成绩；该复核仍属于待完成项。

## 08:55：半小时内存趋势及Decode主进程只读排查

新增离线报告器`tools/report_soak_memory_trends.py`，保存每个PID的完整窗口统计、
输入日志快照hash、RSS区间/线性斜率、FD及Swap，不自动给出“无泄漏”结论。
`memory_trends_early.json`、`memory_trends_30min.json`均由空闲96读取日志生成，未调用GPU。

10～30分钟窗口暴露角色差异：

- D rank0增加68.438MiB，而D rank1～7各增加约1.59～2.66MiB；
- P rank0增加4.473MiB，其他P rank的增量约0.66～9.61MiB；
- Control增加7.723MiB。此时仍不能把初期高水位或较小斜率当成长期稳定证明。

因此对D rank0/1做了两次`/proc/smaps`快照，不使用ptrace、不暂停线程、不读写进程数据：
`maps_decode_32min/`（目录名为早期标签，实际时间以JSON epoch为准）及`maps_decode_followup_r1/`。
两次间隔298.513秒，D rank0仅增加3588KiB（约3.50MiB），rank1增加256KiB；
均为匿名内存变化，所观测GPU设备映射和大型共享库映射RSS未变。
这个短窗口显示增速已放缓，但还不能证明之后不再增长。

只读源码核查发现xLLM核心FastTokenizer已调用Rust释放函数，不能机械套用Control的修复；
PD接收请求表在handoff时有erase路径。BRPC现场控制器数量为6、IOBuf约8.16MiB，
没有看到对象数量积压到已完成请求总数的证据，但单点值也不足以排除其他生命周期问题。
目前没有据这些猜测修改生产代码、重编或重启服务；继续以小时级曲线验证是否收敛。

相关原始证据：`decode_rank0_bvar_readonly_r1.txt`、
`decode_rank{0,1}_mapping_delta_r1.json`及上述smaps快照。
截至约39分钟，914/914严格通过、没有告警，完整5小时验证继续进行。

## 09:20：首小时检查点

压力客户端1159406及监督1159404持续运行，同一批17个服务进程未重启。
首小时后的B1探针（round25）16/16严格正确、实际cached=0：
TTFT2160.593ms、TPOT8.7420ms、92.181tok/s。此前完整B16轮仍约432～433tok/s。
至此1504个已完成请求均严格正确，未见监测告警；仍不是5小时最终验收。

新增`independent_partial_check_r1.json`已从原始输入、SSE和完整正文复核早先1246条，
1246/1246通过、cached=0，无遗漏或文本/流式归档不一致。明确标记运行中部分快照。

`memory_trends_hour1.json`包含全部rank，不自动判定“无泄漏”。30～60分钟窗口：

| 进程范围 | 窗口RSS增量 |
|---|---:|
| P rank0 | 3.660MiB |
| P rank1～7 | 各0.785～1.641MiB |
| D rank0 | 20.121MiB |
| D rank1～7 | 各1.707～2.309MiB |
| Control | 0.980MiB |

D rank0的拟合增速由10～30分钟约222.6MiB/h降至30～60分钟约34.7MiB/h，
Control由约23.3MiB/h降至约1.84MiB/h。增长确实在放缓，但D主进程仍需继续观察，
不能把短窗口外推值或较小增量解释为已证明长期无增长。所有样本Swap为0，进程身份稳定。

## 09:28：输入进入第二轮复用

同一压力进程继续运行，独立部分快照`independent_cycle2_entry.json`复核1697条已完成记录，
1697/1697严格通过，全部实际cached=0。仍有正常在途请求，未将它们计为已完成。

原失败用例`stress-0378`的本轮第75次和第1675次请求均严格正确，生成正文SHA均为
`739594e22b7ac3bac854ee45cfa67cb06bf73e1d1f95658be4ad5337e3c5645b`，
没有出现790→890跳号。该证据说明第二次输入复用通过，不能扩大为原第4875次故障点或
完整5小时已通过；继续保持相同binary/config运行，未重启、改参数或resetGPU。

## 09:35：Decode主进程的只读分配器证据

针对先前短窗口增长较明显的三个64MiB对齐区域，在核验PID/exe/start_ticks后，
通过`/proc/PID/mem`只读取glibc 2.35分配器头和空闲链表元数据，不读取/输出模型数据，
不使用ptrace、不暂停线程、不写进程内存。单次读取约1～4ms；本次三次均取得有效结构，
头部前后相同、mutex前后均为0。此前旧实验中的不一致/超时观察仍保留为inconclusive，未改判。

| 分配器区域 | arena管理空间 | 普通空闲空间 | 存活量上界估计 |
|---|---:|---:|---:|
| 0x7fb9f0000000 | 5.680MiB | 5.368MiB（94.50%） | 0.312MiB |
| 0x7fbc40000000 | 7.031MiB | 6.713MiB（95.48%） | 0.318MiB |
| 0x7fbdc4000000 | 6.887MiB | 6.523MiB（94.72%） | 0.364MiB |

这里的比例是所选arena管理空间，**不是整个进程RSS中空闲比例**；存活上界还包含
allocator元数据、fastbin/tcache等未单独扣除部分。无锁读取也不能替代完整分配跟踪。
这些结果支持“部分增量来自分配器保留/高水位”，但只覆盖所选区域，不能证明全进程无泄漏。
没有调用malloc_trim、更换allocator或清缓存来压低RSS。

原始证据：`decode_rank0_arena_readonly_r1.json`、
`decode_rank0_arena_{7fbc40000000,7fbdc4000000}_r1.json`及`maps_decode_allocator_followup/`。
约1小时20分时1893/1893严格正确，压力进程1159406与监督1159404持续存活，无告警。
继续完整5小时及后测，不将上述局部观察当作最终稳定性验收。

## 09:47：同一arena复查与连续运行检查

约12.6～14分钟后，再次读取同一三块arena，PID/exe/start_ticks保持一致，
结果保存在`decode_rank0_arena_{7fb9f0000000,7fbc40000000,7fbdc4000000}_r2.json`。
三块管理空间大小均未扩大，仍为5.680、7.031、6.887MiB；存活量上界估计分别约
0.354、0.316、0.384MiB，前后头部均一致，读取耗时各约2～3ms。
这支持所选区域以保留空间为主，仍不能代表全部arena或证明全进程长期无增长。

压力进程和监督进程仍为1159406/1159404，约1小时32分钟时已完成2201条、严格通过2201，
无告警；最新完整B16轮433.495tok/s、64/64正确。服务、配置和负载未变，未resetGPU。
完整5小时、后测准确率和老化后同协议性能复核继续列为未完成项。

## 10:20：第二小时内存窗口

`memory_trends_hour2.json`覆盖负载开始后约60～120分钟，三角色均有完整样本、身份稳定、Swap=0。
大多数rank已进入很小斜率区间，但仍保留两个明显阶段性增长，不能用平均值掩盖：

- P rank0：窗口端点增加5.160MiB，拟合斜率5.44MiB/h；P rank1～3、5～7端点增加0.090～0.266MiB；
- P rank4：窗口端点跳增49.418MiB，拟合斜率61.62MiB/h，需要后续继续观察是否为一次性缓存高水位；
- D rank0：窗口端点增加27.684MiB，拟合斜率34.43MiB/h；D rank1～7端点增加1.36～2.37MiB；
- Control：窗口端点增加0.660MiB，拟合斜率0.81MiB/h。

P rank4 和 D rank0 的增长没有伴随OOM、Swap、严重日志或严格输出错误。
监测器并不按RSS增长斜率自动判失败，因此“无安全告警”不能代替内存稳定性判断。
它们不构成“内存问题已解决”的证据，也不自动构成新的根因；需结合第三、四、五小时窗口，
以及arena/映射快照确定是一次性workspace/cache增长还是持续增长。
第二小时报告生成时的进度快照为2942/2942严格通过，最新完整B16约433tok/s。

第三小时资源报告`memory_trends_hour3_final.json`已保存，覆盖约120～180分钟，
全部角色均有完整样本。该窗口中P rank2/rank7各出现约55MiB阶段性跳增，
D rank0该窗口端点增加15.480MiB（之前写成约28MiB是误用了第二小时值，现更正）；其它rank斜率较小。异常rank全部保留，不用均值隐藏，
也未触发OOM/Swap/错误日志。后续小时继续观察这些跳增是否稳定在高水位。

第三小时B1探针16/16严格正确、实际cached=0：TTFT2159.656ms、TPOT8.7794ms、
91.815tok/s。至约3小时04分，4425/4425已完成请求严格正确，最新完整B16约432～433tok/s。

## 11:20：第三小时内存窗口与异常rank保留

`memory_trends_hour3.json`覆盖负载开始后约120～180分钟。三角色进程身份和Swap仍正常，
但RSS增长继续呈现rank不均匀现象，必须保留原始rank数据：

- P rank0窗口增加1.672MiB，斜率2.06MiB/h；P rank1、3～6均低于0.6MiB端点变化；
- P rank2窗口增加55.254MiB、rank7增加54.453MiB，拟合斜率约78MiB/h，属于需要继续追踪的阶段性跳增；
- Control窗口仅增加0.121MiB，斜率0.13MiB/h；
- 这份早期快照中Decode最新样本仅到179.807分钟，报告器因此暂未输出完整窗口；
  不是监控丢失样本。后续`memory_trends_hour3_final.json`已补齐：D rank0增加15.480MiB，
  rank1～7增加0.332～0.953MiB。

当前不能用“多数rank稳定”抵消P rank2/7的跳增，也不能直接把它判为代码泄漏：
前面同类跳增在后续窗口可能是workspace/cache高水位，需和arena及smaps证据结合。
报告生成时长测进度为4377/4377严格通过，未resetGPU；继续执行第四、第五小时。

## 12:30：第四小时窗口与原失败边界的独立审计

仍为相同Control/P/D进程及binary，没有修改运行参数或重启。压力客户端1159406、监督1159404
均存活，三角色监测心跳正常。第四小时B1探针（round96）16/16严格通过、cached=0：
TTFT2159.804ms、TPOT8.7126ms、92.353tok/s。完整B16轮约432～433tok/s。
混合负载从开测起累计约412tok/s包含小时B1探针和请求间隙，不能把这个值称作B16独立吞吐。

`memory_trends_hour4.json`读取完整180～240分钟窗口：

| 进程 | 本窗口RSS端点增量 |
|---|---:|
| P rank0 | 1.047MiB |
| P rank1 | 21.367MiB（窗口末段跳增） |
| P rank2、rank7 | 分别0.430、0.449MiB |
| P rank3～6 | 各0.047～0.113MiB |
| D rank0 | 10.910MiB |
| D rank1～7 | 各0.367～0.727MiB |
| Control | 0.113MiB |

此前P rank2/rank7第三小时的约55MiB跳增在本窗口未持续同速增长；D rank0依次由
第二小时27.684MiB、第三小时15.480MiB降到第四小时10.910MiB，增速继续放缓。
但P rank1出现新阶段性跳增，D rank0尚非零增长，仍需第五小时及空闲尾段验证，不提前宣布内存问题完全验收。

独立快照`independent_after_failure_boundary_hour4.json`确认6115/6115已完成请求严格正确，
实际cached=0，冻结输入/原始SSE/正文/用量均一致。
原失败用例378在第75、1675、3275、4875次请求均正确，正文SHA同为
`739594e22b7ac3bac854ee45cfa67cb06bf73e1d1f95658be4ad5337e3c5645b`。
这证明最终候选在本轮跨过旧失败点；不据此单独认定偶发跳号根因已被完全证明，也不拼接旧版本长测时长。

后续顺序：第五小时排空→自动全量GSM/重复题/90秒空闲尾段→独立官方评分与逐题比较→
新监测窗口下同进程B1/B16性能复测→完整内存、输出与hash证据归档及最终完成审计。

## 12:45：第五轮输入复用通过，保留晚期内存对照

压力客户端1159406及监督1159404仍为原进程，未重启任何服务或修改参数。
`independent_cycle5_entry.json`从冻结输入、SSE和完整正文独立复核6510条已完成响应，
6510/6510严格通过，全部实际cached=0。旧失败用例378在本轮第75、1675、3275、4875、
6475次请求均正确，正文SHA均为
`739594e22b7ac3bac854ee45cfa67cb06bf73e1d1f95658be4ad5337e3c5645b`。
该文件是明确标记的运行中部分快照，不把尚在途请求或余下约半小时算入已完成。

第五小时前约22分钟的只读资源样本中，Control RSS相对240分钟附近未增加，D rank1～7各增加
不足0.3MiB；P rank1仍增加约20MiB、D rank0增加约6.7MiB，二者继续列为内存待验收项。
94保存`maps_prefill_late_r1/{1696439,1696440}.json`，分别对应P rank1及较稳定的rank2，
供负载结束时做同进程映射对照。该操作只读/proc元数据，未暂停线程、写进程内存或调整allocator。

截至约4小时29分客户端进度6510/6510正确、无告警；当前不得提前认定完整5小时、压测后GSM8K
或老化后性能已通过。全量后测与重复探针由现有driver自动发起，最终B1/B16复测仍需后测评分通过后另行启动。

## 13:21：完整5小时负载与排空通过，后测运行中

### 完整负载结果

`soak_final_r1/stress/complete.json`确认：

- 08:15:34开始，5小时提交窗口截止13:15:34，最后请求约13:16:01完成；
- 实际持续18026.601秒（含约26.6秒排空），submitted=completed=strict_passes=7312，failures=0；
- `completed_duration=true`、`soak_pass=true`、stop_reasons为空，客户端正常exit=0；
- 独立`independent_output_audit.json`复核7312条全部冻结输入、原始SSE、完整正文、usage和结束原因，
  无遗漏提交，全部cached_tokens=0；这是完整审计，不再是部分快照。

5轮小时B1探针各16/16正确，TPOT依次8.7907、8.7420、8.7698、8.7794、8.7126ms。
113个B16完整轮各64/64正确，实际吞吐范围430.474～433.819tok/s；
B16各轮实际输出总量/各轮公共时段之和为432.195tok/s。
整轮包含B1和间隙的混合平均为415.358tok/s，不能替代正式同输入配对的4.6975倍结果。
原第4875次跳号未出现，旧case378的五次复用正文hash均一致；仍不将这一结果当作独立根因唯一性证明。

压力客户端1159406正常结束；监督1159404继续运行，自动后测PID1209539已实际发送请求，
截至本次观察完成前244题，未完成全量评分。Control/P/D仍是预验收那17个进程，未resetGPU。
后测完成后还有重复题5次、90秒空闲尾段、官方评分、逐题回归比较及另行启动的同进程性能复测。

### 第五小时RSS与离线统计边界修正

旧报告器在最终时段使用“最后负载期采样时刻≥300分钟”判定窗口完整，
但P最后负载期样本位于299.697分钟，下个分钟样本已属于后测，造成它漏列第五小时。
保留旧输出`memory_trends_load_finished.json`；新离线报告器`report_soak_memory_trends_r2.py`
改为同时核对实际负载已覆盖窗口终点、监测心跳已跨过终点，且窗口内相邻及边界采样间隔均≤90秒。
**统计仍只用240～300分钟内且不晚于负载结束的原始样本，不插值、不补造、不混入后测。**
6项CPU边界测试通过，拒绝未完成负载、过期监测、缺样大间隔、重复样本、窗口外样本；
该修改不涉及服务binary、监控器、原始日志或验收阈值。每个角色第五小时均实际采到60点。

`memory_trends_load_finished_r2.json`第五小时窗口结果：

| 进程范围 | 窗口RSS端点增量 |
|---|---:|
| P rank0 | 2.289MiB |
| P rank1 | 20.570MiB |
| P rank2～7 | 各0.027～0.070MiB |
| D rank0 | 15.281MiB |
| D rank1～7 | 各0.254～0.723MiB |
| Control | 0.003906MiB（4KiB） |

P rank1两次晚期smaps快照间隔1316.8秒，RSS仅增加172KiB，rank2增加4KiB，
大规模阶段性跳增没有在该后续窗口持续同速发生。证据为`maps_prefill_late_r1/`、
`maps_prefill_late_r2/`及`prefill_rank{1,2}_late_mapping_delta.json`。
D rank0第五小时仍有增长，不能只因幅度较小就判定整个内存目标已经完成；
需结合后测和空闲尾段、allocator保留空间证据作最终判断。

当前关键证据SHA256：

- 完整负载complete：`22a4a7a6a8c3e5358ccd32d084cae63957c3c8f278ee31b879e9987a3f20c143`；
- 完整独立输出审计：`48aa2c5915eb254c3e7e2c9db74d8852fa250b2a254ac12f6737743613e9ffcc`；
- 最终负载内存报告r2：`c3dc232bc4d67f6fc558642a33be44b919112d94dfba64129658a5d3c39728d2`。

结论仅为“最终组合完整5小时严格负载通过、负载中性能保持”。Goal整体仍未完成，
尚不能以压测前96.66%代替压测后准确率，也不能以小时探针代替后测后的正式64题性能复测。

## 13:28：完整负载的监测覆盖独立审计

`independent_load_monitor_coverage.json`对已完成负载的首尾时间做独立区间检查，
确认每个角色均有负载开始前及负载结束后的有效样本，覆盖全部17个服务进程；
原始样本逐条核验PID/exe/start_ticks一致，不只检查ready文件或最后一次心跳。

| 角色 | 覆盖负载并包含边界的样本数 | 最大相邻间隔 |
|---|---:|---:|
| Prefill | 301 | 60.427秒 |
| Decode | 302 | 60.562秒 |
| Control | 303 | 60.061秒 |

覆盖期间无严重日志记录、GPU查询错误、Swap或证据盘低空间告警，进程身份稳定。
审计保留读取时的完整样本前缀hash；文件仍继续记录后测，因此不把前缀hash误称为最终文件hash。
此结果只证明采样覆盖及记录到的健康状态，明确`memory_leak_freedom_proven=false`，
最终内存结论仍需结合RSS和分配器证据。

后测客户端1209539与监督1159404持续运行；新建了空的`final_ffi_pd_r1/post_performance_r1/`
用于后测评分通过后的同进程性能复核，**尚未启动复测、没有并发追加任何推理请求**。

## 13:49：压测后全量精度保持，启动同进程正式性能复核

### 全量后测与重复题

后测1319题全部完成、传输错误0。原SGL MathEvaluator评分为**1275/1319=96.6641395%**，
辅助numeric-boxed为1254/1319（95.0720%），均与本次压测前相同，未更改协议或过滤题目。
119、434仍是基线原有两道1024上限截断题，仍算错；没有新增截断或空输出。

本次最终候选直接pre/post比较（不只与旧发布基线相比）：

- 1319个输入payload和参考答案逐题一致；
- symbolic_correct、finish_reason逐题一致，空输出前后均0；
- 1308题生成正文完全相同，11题正文不同，但评分和结束原因不变；不声称这些正文逐token完全一致；
- 1077后测返回145、164tokens、stop，随后1077/1077/1075/1076/1077重复穿插5条均正确，
  其中1077的3次重复均cached=128、正常stop，未出现旧版只输出We或重复续写。

评分和对照位于`soak_final_r1/gsm_after_official/logs/pd_final_gsm1319_after_20261007/`，
包括summary、全部scored行、`baseline_comparison.json`、`pre_post_comparison.json`。
后测评分输入SHA：`550881434168f9d30f015eab6145106316e62d2cabb046fbd5fc988c9e124308`；
scored JSONL SHA：`696c6c5d26c2159d48543f2c5c70e9057150b8b7f5af1148a0e6bce915836620`；
完整评分包SHA：`a4bf62d6d5202a237999cccf4c3f39eba130f415da543ff767621ba7c31fb62d`。
`orchestration_complete.json`保留生成器原始的official_gsm_scoring_pending=true；
它只表示生成器不负责评分，以上单独官方评分结果已经完成，不回写/覆盖原始记录。

### 空闲尾段与监测窗口轮换

`idle_tail.json`记录90.099秒自然空闲（1791351700.7216532～1791351790.8203502）。
该窗口中D有两个样本：rank0增加16KiB，rank1～7不变；Control两个样本RSS不变。
P只有一个落在窗口内的周期样本，不能用该窗口单独拟合P的增长或断言其零增长。
没有malloc_trim、清缓存或重启来人为降低RSS。

原负载/后测driver1159404和客户端1209539均正常结束；P/D监测按完成标记退出，
Control通过本地完成标记结束旧监测及桥接，仅轮换监测器，不停止服务。

### 老化后的正式B1/B16复测已运行，尚待结果

复测目录`final_ffi_pd_r1/post_performance_r1/`，驱动PID1214380，
命令脚本`preacceptance_tools/run_post_performance.py`，SHA保持
`9b14364b8ae30d844f0803e92f29d37b626f107df7aff243430294c908c67254`。
开启新监测：D PID1214208、P PID1767800、92宿主机Control桥接PID3229601，
Control本地源目录`/tmp/pd_control_tokenizer_20261007/monitor_final_postperf_r1`。
脚本已核验Control/P/D的PID、exe、start_ticks与原长测完全相同，禁止重启后的成绩冒充老化复测。

使用同一304～319预热、320～383正式输入，每档64条正式请求，要求实际cached=0、
严格全文正确、B1 TPOT<9ms和B16实际吞吐倍率≥4.5。结果前缀`final_ffi_pd_r1_aged`。
目前只观察到预热阶段正常响应；不能提前把预热或原先性能成绩当作压测后正式结果。

## 14:14：老化后正式性能通过，剩余RSS的广覆盖只读检查

`post_performance_r1/post_performance_complete.json`已完成，所有阶段exit=0、alarms为空。
驱动1214380正常退出，模型服务继续保持运行；结束的仅是资源监测器和评测客户端。
`post_performance_r1/independent_pair_audit.json`独立核对同一组64条冻结输入、完整正文、
真实cached=0、实际完成token数/完整客户端时间以及前后17个服务进程身份。

| 同一最终组合，30K/1024冷缓存 | 压测前 | 压测及全量后测后 |
|---|---:|---:|
| B1严格正确 | 64/64 | 64/64 |
| B1 TTFT | 2183.681ms | 2159.620ms |
| B1 TPOT | 8.7546ms | 8.7708ms |
| B1聚合输出吞吐 | 91.856tok/s | 91.862tok/s |
| B16严格正确 | 64/64 | 64/64 |
| B16 TTFT | 14804.504ms | 14812.460ms |
| B16 TPOT | 19.8648ms | 19.8392ms |
| B16聚合输出吞吐 | 431.494tok/s | 431.850tok/s |
| B16/B1吞吐倍率 | 4.6975 | 4.7011 |

B1 TPOT变化+0.185%，B16吞吐变化+0.082%，属于该实验下很小的波动，不宣称额外加速。
两项性能目标均通过；B16数值是并发上限16的公共时段聚合吞吐，不是16/TPOT推算。
前后配对输入SHA仍为`ad076bca62687afe22844e0ae784045c90f042c314a4aa2bd4b5638ee8886be6`。

### 不干扰性能测试的额外内存检查

正式复测退出后，才读取D rank0/1最终空闲smaps，保存在`post_performance_r1/maps_decode_final_idle_r1/`。
与负载约77分钟时的既有D rank0快照相隔16452.4秒，RSS增加66948KiB，均来自匿名映射；
这个区间包括剩余负载、GSM后测和性能复测，不能误称为纯负载窗口增长率。

`inspect_grown_arenas.py`按正向增长大小排序，仅对最大96个、64MiB对齐的匿名可写候选区域，
调用已验证的有界glibc元数据只读助手。总耗时4.686秒，未使用ptrace、暂停、内存写入、trim或reset。
96个均取得一致的无锁头/空闲链表快照，覆盖66124/66948KiB（98.77%）正向RSS映射增量。
剩余178个小候选区域及其它映射的824KiB增量未被这组元数据检查覆盖，不能省略该限制。

其中单区域增长≥128KiB的44个区域贡献65124KiB（97.28%）增量，当前arena管理空间合计
294563840 bytes，普通空闲空间270069232 bytes（91.68%），存活量上界24494608 bytes。
最大的增长区域各自仅约0.3～0.4MiB存活量上界，却保留约7～9MiB管理空间。
所有96个区域的总量也保存在summary中，未用筛选后的比例冒充全体或整个进程比例。

证据：`grown_arenas_readonly_r1/summary.json`及逐区域原始记录，
`decode_rank0_final_mapping_delta.json`。结论支持“主要剩余增量所在区域存在大量已释放但保留的空间”，
与已修复的FFI对象引用泄漏应区分；但普通空闲字节不等于空闲驻留字节，
单时点的存活量也不能证明所有区域历史存活量没有增长。保留
`proves_whole_process_leak_freedom=false`，不宣称无限时长全进程绝无泄漏。

性能、完整负载、GSM后测及重复题验收均已完成；接下来封存独立源码/binary/config/input/log/result/hash，
完成最后逐项审计后再决定Goal完成状态，避免把尚未收尾的证据归档当作已交付。

## 14:41：完整证据封存与额外有界内存对照

已经封存 `final_sealed_evidence_20261007/dsv4_pd_stability_evidence_20261007.tar.gz`，
大小317562156 bytes，SHA256为
`3be71730237d424787875ae40e489bfb61df708efc8cc2305b925ef30bad1767`。
重新打开压缩包逐文件校验11120项，原始快照总量3101046196 bytes；
内嵌与外置manifest SHA256均为
`7c78117aaa43865a9ec8cf7883570c5ecfc30897008b2126c62e382d9862fedb`。
原始实验目录不删除、不覆盖。仍在写入的服务日志按封存时大小保存前缀，并注明截止点。
包含源码覆盖层、实际P/D/Control二进制与增量构建证据、冻结输入、配置、全部验收输出和监测记录；
模型权重及外部AOT依赖不重复打包，保留路径和hash，复现仍需要这些外部资产。
本节及后续结论晚于压缩包内文档快照，以包外最新报告为准。

为避免把“空闲arena占比高”错误解释为“没有存活对象积累”，在同一批老化服务上补充
`residual_liveheap_256_r1`有界对照：原冻结客户端、索引400～655、并发16、共256请求，
每请求仍为30K/1024并完整核验512个数字。结束排空后自然空闲30秒，再读取同一组96个arena。
逐一比较system/free/inuse-upper-bound，并检查arena身份、PID/exe/start_ticks；
不修改服务、allocator、模型或推理参数，不注入trim，也不resetGPU。
这是额外诊断，不替换已经通过的完整5小时与正式性能成绩；当前尚待其结果。

## 14:54：最终内存对照与本轮验收结论

新增请求全部完成，独立审计为256/256严格正确、0传输错误、cached=0。
原输入/客户端hash不变，未重启服务；请求用时559.120秒，结束后自然空闲30秒再读取。
本组不是之前320～383配对性能输入，不用它的468.851tok/s替换正式B16的431.850tok/s。
冻结客户端summary中的历史“8GPU”标签不适用于PD；实际仍是P8+D8、共16卡。

同96个唯一arena的前后无锁一致快照：

| 指标 | 请求前 | 排空后 | 变化 |
|---|---:|---:|---:|
| arena管理空间（bytes） | 2510151680 | 2512842752 | +2691072 |
| 普通空闲空间（bytes） | 372922976 | 375433648 | +2510672 |
| 存活量上界（bytes） | 2137228704 | 2137409104 | +180400 |
| D rank0进程RSS（KiB） | 9857040 | 9859676 | +2636 |

管理空间增量的93.3%已在普通空闲链表中；存活量上界变化仅约176.2KiB，
约为本组原有上界的0.0084%。上界仍包含allocator metadata、fastbin、tcache，不能等同纯业务存活对象。
这组结果支持剩余增长主要为分配器保留，而非旧版FFI对象每请求持续泄漏；
不能把空闲字节比例直接当作空闲RSS比例，也不覆盖所有未选中的arena。
此后3次空闲快照覆盖同96个arena，管理空间均为2512842752 bytes；
存活量上界依次2137420608、2137423808、2137426064 bytes，49.58秒变化5456 bytes。
未用trim/reset/重启降低RSS；空闲期间少量变化仍保留，不写成零增长或无限期无泄漏证明。

`final_acceptance_audit_20261007.json`汇总并hash关联以下验收，全部通过：

- 完整5小时及排空、7312/7312严格输出、原失败378在5轮历史位置均正确；
- 同17个服务进程，无中途重启，GSM8K前后均1275/1319=96.6641%，逐题评分/结束原因无退化；
- 1077等重复穿插探针5/5正确；没有新增截断或空输出，原119/434仍计错；
- 正式老化后30K/1024：B1 TPOT8.7708ms<9ms；B16实际公共时间吞吐431.850tok/s，
  是B1 91.862tok/s的4.7011倍；B1/B16均64/64严格通过；
- 额外256请求及同arena对照通过，主要所有权泄漏已有独立CPU负例/正例与单机GPU对照证据。

本轮修复在既定5小时稳定性、正确性和性能范围内完成验收。偶发长计数跳号在新组合中未复现，
但不能宣称其唯一根因已被单独A/B证明；不把计数/GSM8K验收扩展成所有长上下文任务的保证。

主要代码变更仍为Decode自定义prefix-state producer guard、MoE/FP8 FFI tuple拥有型参数、
Control HTTP protobuf Arena生命周期和Rust encode结果释放。模型、AOT、量化数学、
Graph/Overlap/Prefix生产配置未为本次验收降级。沿用增量编译依赖，未更换Mooncake。

追加证据独立保存到` supplemental_evidence_20261007/residual_heap_and_final_acceptance.tar.gz`（路径无前导空格），
包含256请求、前后/空闲arena快照、最终验收表、工具和最新报告；通过其`complete.json`查看SHA256。
原5小时压缩包不覆盖。Windows独立工作树为
`C:\Users\jiaming.wang\code\xllm-musa-pd-stability-20261007`，
分支`bugfix/dspark-pd-stability-20261007`，当前修改尚未commit/push；未改其他远端分支。
Control修复源码另在证据包内，不应只复制xLLM二进制而漏掉Control修复。
服务继续保留在94P/90D/92Control；测试客户端已结束，全程无GPU reset。
