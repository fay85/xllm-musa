# Decoder 尾段 Graph 三组正式测试

先决条件是 P 端计算流独立诊断成功且完成正式验收，禁止在其尚未解决的停滞上叠加 Graph 后混淆原因。

使用同一 cumulative ELF，当前累计优化全保留、Gate 缓存关闭。依次 OFF、GRAPH6（6次输入复制）、GRAPH1（融合输入复制）三组；只在 D 端打开 cache-free decoder 尾段 Graph，Attention/缓存仍为 eager。每组两道算术题、256/256及2048/256各1暖5正式，再独立抓同 q1/kv263 的16rank trace，正常停止全部我方角色后才启下一组。结果按实际 token/success核验。

已通过40-generation八rank共享pool IPC Graph 基本通信检查，不代表整个模型已经通过。若 capture/replay失败，不捕获异常后回退 eager 继续请求；保留故障、正常 TERM、停止自动重启。只有完整数据和 trace 得出的接受/拒绝结论才更新累计最优。

仅通过基础功能检查，未完成模型精度验收。
