# T14：原生 draft factory 与真实 checkpoint 接线

该版本用显式 XLLM_DSV41_DSPARK_NATIVE 开关选择新原生 V41 DSpark，默认仍不启用；其余模型和旧 V4 draft 不改路由。V41 draft 保持ratio0、3层、128专家/top3；不能继承目标40层/384专家/top6或将ratio0归一化成旧V4的ratio1。

接线复用已验证的 FP8 dense 解码和 INT4/group128 专家加载，mtp.0 main_proj/main_norm、各mtp decoder、mtp.2 norm、共享词表头，以及实际checkpoint命名的markov_head.embed/head和confidence_head.proj。已有通用 Markov 类读取markov_w1/w2，不能直接假设本模型键名相同。

当前尚未开放完整 speculative serving：还需要target aux entering-residual捕获、压缩器/Engram/SWA状态验证回滚与runtime接入，并经实际加载和基本问答验证。此构建成功不等于MTP完整实现或性能晋级。

仅通过基础功能检查，未完成模型精度验收。
