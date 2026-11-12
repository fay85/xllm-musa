# T12：支持B=2/4的真实多请求Decode，不将HTTP排队称为批处理

现有Native在forward入口要求num_reqs==1，模型循环将q>1当作同一请求的Prefill，不能直接去掉CHECK。GPU数学大多已经接受多行，但缓存、不同上下文长度、Engram与CSA2 source/runtime必须按请求隔离。

计划保留P端C1、D端新增显式B=2/4 decode路径：从host cumulative q/kv长度读取每请求q1/context；embedding、mHC、QKV投影、FFN/路由/专家GEMM按B行合并；Attention投影以后按各请求的页表、positions、context和独立CSA2 runtime组织cache/index/压缩状态，再将每请求attention结果拼回B行，统一output projection与归约。Engram hash按各请求lookback分别形成ids，lookup/injection再合批。P不混入不同请求Prefill，D不接受q>1的混合batch，先把范围做清楚。

只要Attention仍逐请求准备，就如实说明这是投影/FFN合批而非所有attention metadata完全融合。不能把C2串行队列吞吐冒充B2 GEMM收益。大于1行时现有TP/EP调用选择可能退回MCCL，需核对或扩展custom适用范围并做8rank小消息检查。

基本回归应包括两个不同页表、上下文、Engram lookback和source/topk状态的请求；验证与分别C1执行的各自结果、未触碰别人的cache，以及batch成员变化/完成/新加入。模型快速验收仅两道计算题；浮点或文本差异按用户口径记录，不展开精度调查。正式C1两组仍保留，C2/C4吞吐与尾延迟作为新场景单列，检查每请求真实输入输出token和成功状态。

该文件为待实现设计，不表示已经支持真实B2/B4。
