# GLM-5.3-Flash FP8 MegaMoE 双机1P1D实验记录

本目录对应共同仓库个人实验分支 xiaopang1104benchmark1P1D。
被测代码由 provenance/source_manifest_v14c.json 锁定；native二进制、模型与依赖哈希另见provenance。
当前按用户要求保留已知精度问题并记录性能，不新增分歧复测或GPU trace，不创建MR/PR。

- [详细报告](adaptation_report.md)：结构、实现、数值证据、运行配置与限制。
- [接入状态及0062/0065说明](当前接入状态_0062与0065及数值边界.md)：MATE与我方GPU实现的边界，严格135Token前缀反例的具体位置。
- [性能结果](性能结果.md)：两组输入输出、官方EvalScope统计与原始证据口径。
- [题面与标签疑点](MathQA题面与标签疑点说明.md)：冻结标签不改，数据集问题与模型错误分开记录。
- [协议与复现入口](结果口径与复现入口.md)：请求冻结、0071恢复与Token校验规则。

accuracy/official_initial100保存官方初次100题；accuracy/full_pd_fp8_v14c保存本地初次100题、
0071原始失败和恢复、已有官方追加记录及逐题汇总。追加记录不覆盖主分数，未完成的本地三次复测不伪装为已完成。
performance保存正式EvalScope原始SQLite、SSE消息、逐请求native完成行及独立暖机。
diagnostics保存历史数值对照与运行日志；图目录中的数值图/准确率图均不是新GPU trace。
reproduction/original_script_text保存原样实验脚本文本及其哈希，作为实验记录，不是仓库安装入口。

本报告的中文原文来自AI Book本轮目录，文中的“评测/日志/参考材料”对应本目录的
accuracy/performance、diagnostics和provenance；NFS完整实验目录为
/data/nfs_shared/xiaopang/20261105_glm53_full_adaptation。
权重、内部库完整源码、密钥和大型二进制不进入本提交。
