# T14 实机缓存合同修复

E44首次完整PD请求在P端write_context_kv失败：真实值[21,512]，缓存[21,576]。8个P rank主动CHECK退出(-6)，D端收到正常TERM后8rank全部exit0，两台GPU资源已释放，无SIGKILL。根因定位到create_dsv4_cache_tensors只把deepseek_v41识别为V4.1缓存；dedicated drafter名为deepseek_v41_dspark，被落入V4 FlashMLA路径，额外分配64维RoPE副本。

仅在MUSA设备缓存创建时把V4.1 DSpark归入V4.1缓存分配分支。专用drafter已清空KV source层、compress ratios为0，因此只分配SWA512；不伪造64维零填充、不删尺寸检查、不裁掉真实计算。V4仍保留576布局，非MUSA逻辑保持原有选择。当前仅准备源码，需编译、真实P/D再验证，不能提前宣称问题解决。
