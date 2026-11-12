# T02a：复用当前 decode forward 的只读压缩缓存

**接受为累计速度基线。仅通过基础功能检查，未完成模型精度验收。**

本项完成的是T02的压缩缓存重复gather子项；SWA读侧页地址与metadata融合仍待继续，不把T02整个项目写成全部完成。

## 实际结果

SJ9(P)/SJ10(D)，各8卡，TP8/EP8、eager、C1、chunk256。保持D端已验收的TP/EP custom，P端保持原路径。官方EvalScope各1暖+5正式。开/关使用同一ELF `3e54a4bb6c4ccfd9e5cc5e298a82f6cea7c173019911592c436e80763295b2c7`，只改变D端 `XLLM_DSV41_DECODE_CACHE_READ_REUSE`。

|输入/输出|缓存复用|TTFT ms|TPOT ms/token|E2E s|输出token/s|QPS|
|---|---|---:|---:|---:|---:|---:|
|256/256|off|357.827|146.529|37.722716|6.786362|0.026509|
|256/256|reuse|466.855|143.904|37.162294|6.888703|0.026909|
|2048/256|off|2367.071|148.582|40.255587|6.359366|0.024841|
|2048/256|reuse|2401.132|144.102|39.147154|6.539428|0.025545|

TPOT下降1.79%/3.02%，端到端输出吞吐提高1.51%/2.83%。所有20次正式请求均成功，输入分别256/2048、输出256；请求内容哈希一致，profiler关闭，暖机不算正式结果。

**TTFT取舍：** 短输入平均357.83→466.86ms，回退109.03ms（30.47%）；长输入2367.07→2401.13ms，回退34.06ms（1.44%）。P端未改，保留所有样本，不删除较慢点，也不声称本项改善TTFT。此前已经出现约200ms的客户端/服务端差值，原因留到T07定位；当前按用户优先关注TPOT与整体输出速度的口径接受，若后续要求TTFT不回退，应单独复核这一取舍。

## 改动与边界

Full层发布的压缩Key与Indexer key，在当前forward、当前来源层内保持只读。以前后续Reindex/Reuse层各自重新从物理页gather；现在Lazy ReadView只materialize一次，后续消费同一Tensor。每次Full publish清空read view；新的forward创建新的Runtime，不能跨token、请求、P/D导入复用。Query、score、Top-K、候选集合仍按原层规则计算，未把不同层的注意力答案缓存起来。

`token_cache.h`新增只读view，`attention.h`管理其生命周期和decode开关，原生模型配置透传，测试绑定暴露真实C++类。Prefill保持原路径；模型计算仍在MUSA GPU，未加入CPU模型计算fallback。

## trace与直接相关的检查

前后各16rank的文件大小/SHA、40层、真实C++ RecordFunction/回溯关联均通过。所有D端rank在选定q=1、kv=263中，compressed_cache_gather从46降到8；8个materialize对应4个Key和4个Indexer来源。D0 CPU index_select 335→259，arange 336→260，where 450→374。选中CPU时间窗口内完整GPU kernel 5654→5246；这个窗口计数不等于完整异步step的精确总数，原始trace保留窗口外关联事件。

所以本项证据是减少重复页映射、gather、mask及其launch，而不是更换GEMM精度。调用数减少82.6%不能当作全模型加速82.6%；严格关键路径可消除占比与理论收益兑现率仍未知。

32个真实C++/GPU cache检查通过：宽128/512、页表int32/int64、committed=0/1/127/128/129/255/256/257、padding含NaN、重复读取保持同一storage、新view看见新commit。两模式都答4.2/56。没有运行API/MathQA/固定前缀数值门槛，未完成模型精度验收。

两模式所有模型rank正常退出0，没有SIGKILL、重启或驱动变更。

## 产物

- 正式报告、SQLite、完整原始响应和服务日志：`04_性能/t02_cache_*_v1_raw`，原始压缩包同目录。
- 大trace保留NFS，路径/SHA见 `05_trace/processed_trace_bundle_v1.receipt.json`。
- 直接检查见 `03_正确性/cache_read_checks.json`；调用计数见 `05_trace/缓存调用次数与关联确认.json`。
- 实际时间线与入口COUNT火焰图见 `06_图/BEFORE_D0`、`06_图/AFTER_D0`。入口COUNT不代表CPU耗时；重profiler窗口不能替代正式TPOT。

后续以本项为母本测mHC输出尾部融合，再推进MoE/Graph和其余T02～T14。仅捕获一个很小的mHC Graph已通过3组输入的算子基本检查，但包含输入copy时微测试更慢，暂不晋级为服务优化。

## 实际图

![正式全部样本](../06_图/两组正式性能全部原始点.png)

![优化前时间线](../06_图/BEFORE_D0/真实CPU区间与GPU时间线.png)

![优化后时间线](../06_图/AFTER_D0/真实CPU区间与GPU时间线.png)

![优化后入口COUNT](../06_图/AFTER_D0/实际C++回溯_ENTRY_COUNT火焰图.png)

缓存相关CPU与GPU细碎调用减少。入口COUNT由540变为510，与gather减少38次、增加8次materialize对应；这个数不是耗时。正式性能只使用profiler关闭的请求。
