# E44：先把完整模型 Graph 故障缩小

E43 owner eager能回答基础题，但完整owner Graph第一道题超时并出现设备故障。故障进程自行退出后、小算例恢复检查通过；没有SIGKILL/GPU reset/重启/改驱动。不能重复原失败配置碰运气。

此前两项证据各有缺口：E38真实专家Graph使用填充行而非非owner真正0行；E43原生40次LL图使用identity专家而非真实MATE专家。本次补齐这一组合：使用官方W4AFP8-mt权重第0层、384专家按8rank分片，top6、owner q4、其他7rank实际0行；捕获stream和回放stream不同。

|检查|实测结果|进程退出|
|---|---|---|
|1次真实专家计算，3种动态输入/路由|8rank×3=24项，Graph/eager逐位一致|8rank均exit0，09:14:05|
|40次连续真实专家计算，3种动态输入/路由|8rank×3=24项，Graph/eager逐位一致|8rank均exit0，09:19:22|

40次使用同一层权重，输出递推`0.01*output + original_input`，保留数据依赖并控制数值范围；**不是完整40层模型，不含Attention或MCCL，也不是性能成绩**。这48项结果说明实际0行与连续真实专家计算组合可以捕获/回放，不能证明原生产Graph安全。

NFS：`/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/diagnostics/e44_real_chain_layer1_v1` 与 `e44_real_chain_layer40_v1`。

下一项增加生产MusaNativeCollectives的embedding allgather（图内）与结果broadcast（图外），保持模型中这一通信边界，继续隔离后再决定生产改动。首版独立测试编译缺少MARCH_TYPE，导致mcclBfloat16声明缺失，已留编译失败记录；它是CPU构建失败，不是GPU运行结果。
