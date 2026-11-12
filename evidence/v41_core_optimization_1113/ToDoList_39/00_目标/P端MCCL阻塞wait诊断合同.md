# P端 MCCL blocking wait 单变量诊断

当前代码已经在TP8=EP8时共用TP communicator，不能把“统一TP/EP communicator”重复当成新修复。已经核对4411与实际c404候选源码的expert_reduction_group，二者都如此。

实验31的P非默认流在第8条2048/8请求停滞，不能晋级。新诊断恢复当前已接受的Pstream0/FFIsync1，D端stream/shared/TP+EP custom/SWA/cache保持累计最优；只在P设置TORCH_MCCL_BLOCKING_WAIT=1。当前安装torch_musa的ProcessGroupMCCL.cpp明确支持此变量，Work.wait在开启时等待通信完成，默认只建立stream依赖。假设是更明确的通信完成边界可以约束异步缓冲区/执行时序；尚未证明是根因。

使用同一c404 ELF，Gate0、Graph0；before引用已经成功的实验26 OFF（Pstream0，blocking wait默认关闭）。先两道算术题、10次无cache2048/8，再两组各1暖5正式及独立trace。不得只靠几次稳定请求宣称解决；同时报告阻塞wait对TTFT的成本。若失败留现场，正常退出和必要的已授权身份核验清理，不自动重启服务或操作驱动。

仅通过基础功能检查，未完成模型精度验收。
