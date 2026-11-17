docs(perf): 我完成 proposal3/5 对照，并补上共享编译环境的测速检查

家明，我在当前E52最优配置上比较了DSpark每轮提出3个和5个候选token。
双机各8卡、实际256输入/256输出、并发1、每组1次暖机后5次正式：
proposal3为21.1979ms/token、44.6403token/s；proposal5为23.0721ms/token、
41.2217token/s；恢复proposal3后为21.1321ms/token、44.4807token/s。
这次没有看到增大proposal带来收益，所以保留proposal3，没有改动生产模型源码。

15条正式请求均成功且实际256/256；服务日志显示verify轮数107→110，
更多候选没有减少验证轮数。独立trace确认target从q4变q6，draft仍是3个学习层；
我修正了trace验收脚本把proposal数误当作draft层数的假设，原失败记录保留。
三组共48个rank均正常退出并释放显存，没有强制清理或操作他人任务。

后续收到SJ9外部编译可能干扰测速的提醒，这几组缺少覆盖正式窗口的连续编译记录，
因此我把它们注明为有潜在CPU干扰的候选观察，不作为最终稳定验收。
已增加SJ9/SJ10编译进程归属观察和测速前检查；外部编译结束后再测，
测试中出现编译重叠的轮次不晋级。外部kill/stop我方任务时先让出20分钟，再检查两机空闲。

当前E52单轮最佳仍为21.0099ms/token、45.0437token/s、TTFT325.8601ms。
E55 draft路由SIMT候选已独立编译，待外部任务结束后实测。
最终同一版本三轮、2048/256正式回归及同窗口SGLang B仍未完成。
仅通过基础功能检查，未完成模型精度验收。

原始EvalScope数据库、服务日志、实际token/接受计数、trace图片及SHA在
evidence/v41_match_sglang_1116/实验054；完整trace在NFS：
/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/profiling/p54_rope_proposal_*。
本次继续提交共同仓库个人分支，不提MR。
