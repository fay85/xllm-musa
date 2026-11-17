# E43 Graph 候选未通过，不能替换累计最佳

## 已证实的现象

同一 ELF（SHA256 `8bfb199226c3712c56b8412ef42133638c3c9d1078ac0ee4a2b279a2899757a4`）的 eager 路径已答对两道题，并完成 1 暖机＋2 条实际 256/256。切到完整 target Graph 后，八个 rank 于 08:48:22 记录 capacity512、q4、parity1 的捕获完成，第一道计算题没有结束；客户端于 08:50:08 达到 read timeout，尚未开始正式性能测试。不能给此 Graph 路径填入性能成绩或宣布正确性通过。

正常退出请求之后，SJ10 GPU3 在 08:51:43 报告 `submission response FAULT`、`DM 4 Run Timeout`、`CSW_TIMEOUT`、`FW_PANIC` 与 `XID=1000007-gpu_run_timeout`，PCI 为 `0000:63:00.0`，报错任务宿主 PID47108 对应我方 rank3。随后 rank3 和 rank0 自行 exit -6。P 端全部正常退出；08:53 快照中 D 端 rank1/2/4/5/6/7 仍等待，未见 D 态。这是设备级异常；没有据此断言硬件损坏，也尚未定位到具体 kernel。

## 操作边界与证据

没有执行 SIGKILL、GPU reset、驱动变更、Docker daemon/服务器重启或 BMC。按用户此前“设备级异常时停止进一步强制操作”的要求，暂停 SJ10 的强制清理与新 GPU 试验，已经请求协调运维/驱动同事确认恢复状态。没有碰他人任务。

- 原始 run：`/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/runs/p43_owner_ll_graph_v1`
- 完整主机日志：该 run 的 `failure/sj10_kernel_since0846.log`，首次归档含 2770 行。
- 进程、容器、实际程序与线程状态：`sj10/device_audit_before_any_force.json` 及本地 `02_构建与短检查` 中相应回执。
- 故障快照包：`exports/e43_graph_failure_snapshot_v1.tar.gz`，SHA256 `7a7f2c7931566585b336f77565f6beb23645433c92101a2ae819a5111a817570`；本地已下载并核验。此包是仍有残留进程时的快照，不能称为资源已释放。

## 已排查与仍未知

在已释放且无本轮设备异常的 SJ9 上，单独扩展原生小测试：40 次连续 LL 通信，真实 0 行非 owner、q1/q4/q6，不同 stream 捕获与回放，6 次不加逐轮 CPU barrier 的连续回放，共336项通过，8 rank 全 exit0。生产模型仍不因此视为通过。

这排除了“跨 stream 或去掉 CPU barrier 本身一定失败”的简单解释。E38 的真实专家检查虽然覆盖真实权重和 Graph，但 true-empty 非 owner 主要在 eager 中检查；Graph 里非 owner 使用固定行数和无效专家 ID。下一步必须补齐真实 0 行＋真实专家＋多层连续 Graph 的组合验证，不能拿 identity 专家或单层检查替代。

另外，生产图还包含 embedding 的 native MCCL gather，回放之后有 hidden/aux hidden 广播；小 LL 测试没有这部分。需要检查 MCCL 与 DeepEP 的 stream/依赖和 Graph 池生命周期，必要时用显式、可观测的布局把 embedding gather 移到 Graph 外，保持同样的计算与通信语义。这是待验证假设，不是已证实根因。

当前不得重复启动相同的完整 Graph 实验。先完成隔离复现和状态确认，再恢复正式对照。已准备的 `p43_owner_ll_off_control_v1` 尚未启动，不得误写为已测速。

## 累计结论

E43 owner eager 是可运行的集成前置；完整 Graph 为失败候选。累计最佳仍为 E39：TPOT27.843298 ms/token、TTFT331.764857 ms、端到端输出34.446540 token/s，尚未追平目标。E43 暂无正式比较、独立 trace 或2048/256通过结论。

**仅通过基础功能检查，未完成模型精度验收。**

## 09:01 资源后续状态

09:00:17宿主机核验所有残留模型rank已经自行退出，09:01:12SJ9/SJ10均无GPU进程。没有使用强制清理或重启。已另行启动SJ10八卡逐卡小算例检查；这不撤销设备故障事实，也不证明原失败Graph组合安全。

09:02:43八卡逐元素与BF16小矩阵乘法全部通过。此结果仅证明基础设备计算恢复，不证明原Graph组合无问题；未复跑原故障配置。
