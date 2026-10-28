# MUSA 大缓存与 RR 固定配置

默认 `VL_PROFILE=large_kv_rr`：Engine 显存比例0.90，Service grant cap及每Router/tenant cap均8192，
credit safety10000；3×CP2 Prefill＋2×TP1 Decode，MTP K2，prefix cache，Prefill piecewise/Decode full graph。
Service 选择独立RR探测游标修复版 `xllm_master_serving.rr_probe_fix_20261024`，业务选路保持RR。
容量与准入仍以真实Engine报告为准，8192不代表可突破实际KV容量。

`VL_PROFILE=baseline` 回退到0.80/cap4096及旧 `credit_fix_20261024_v2` Service。
这只是参数与Service回退；如需复现某次历史结果，还必须显式选择该次Engine二进制。

每项 `VL_` 参数仍可覆盖；`profile_info.sh` 输出有效配置，`stack_info.txt`记录实际参数与二进制散列。
只使用本目录的 `start_stack.sh`，不使用旧 `/tmp` 脚本或 sweep_k0/k1 实验副本。

```bash
bash profile_info.sh
bash start_stack.sh
bash bench_four_rounds.sh 12
bash stop_stack.sh
```

正式性能协议为每档独立冷初始化、4×60完整VL预热、60条新尾部测量，40960输入（含5张854×480图片）、
100实际输出，逐条核验32768缓存token。冻结数据和原始runner由镜像release bundle提供；
`bench.sh`仍是历史短预热诊断入口，不作为四轮资格证据。

历史大缓存＋RR结果为C12=3.385QPS/1150.8ms TTFT、C16=3.587QPS/1764.3ms；
对应旧Engine MD5 `11123f733c0eaf8e849c69365db5dfc5` 和Service MD5 `53824e6f119d95931d2a8ef020012dc9`。
历史Engine已被后续实验覆盖，冻结镜像使用带VMM修复的独立新Engine；不得把上述历史数字标为新镜像实测。

此固定配置默认关闭Graph VMM，保持历史大缓存/RR的开关口径。镜像中的新Engine包含VMM生命周期修复，
可显式设置 `VL_ENABLE_GRAPH_VMM_POOL=true`；该路径已有原生GPU回归，尚不等同于完整VL性能资格。

启动门控使用Service `/readyz`，不把 `/v1/models` 的200当作可接收流量。停止时先Engine，再前端/Coordinator，etcd最后停止。
