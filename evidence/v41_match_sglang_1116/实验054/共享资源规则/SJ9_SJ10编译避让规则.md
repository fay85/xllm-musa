# 性能测试等待外部编译结束

用户于2026-11-17明确要求：以后SJ9/SJ10的外部编译都需等待完成后再进行我方性能测试。与外部kill/stop后的20分钟避让规则并行生效；没有kill/stop事件不能虚构20分钟倒计时。

已加入两机只读进程观察、启动守卫和正式时间窗审计。按实际exe/comm识别gcc、clang、mcc、cc1plus、ninja、make、cmake、链接器等，记录PID/start_ticks/容器归属和CPU负载，不记录完整命令行或凭据，不发任何信号。每10秒采样，连续120秒无已识别编译进程、最新样本不超过30秒且启动时再次检查，才允许测速。采样会漏掉很短的进程，不把它当成独占证明。

NFS证据：`/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/00_control/compilation_monitor/{sj9,sj10}/observations.jsonl`。监控为我方CPU只读任务，正式测试结束后正常停止。各轮`formal_client/compilation_window.json`保存时间窗样本；发现编译重叠或观测缺口则该轮不晋级，等待后另建新轮，不覆盖旧数据。

E54在此规则加入前已完成，缺少覆盖正式阶段的编译进程连续记录。用户提醒时我方两端16rank均已正常退出。其数据仍是真实测量，须注明潜在CPU干扰，不作为最终三轮。后续在受监控窗口复核关键结论。
