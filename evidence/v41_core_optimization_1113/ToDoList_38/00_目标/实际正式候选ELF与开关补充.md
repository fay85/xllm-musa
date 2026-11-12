# 正式测试候选更新

为了保留稳定的P通信边界和累计最优D方案，正式AB使用实验43的c608c4bf… ELF，而非本目录最早的64901cba…候选。实验43是在4411源码上显式整合可选择候选，未验收项全部关闭；完整patch与构建/relay回执在ToDoList_43。当前只切换D端mHC fused reduce，P固定stream0/FFIsync1/通信当前流完成fence1，MCCL blocking wait0。Gate、Graph、packed、CED、batch、DSpark关闭。

先以当前实际module重新做mHC/packed/CED/batch/state/aux capture六组基本检查，再测同ELF OFF/MHC。OFF另做10条长输入稳定性，以确认组合基础；每模式两题、两组各1暖5正式及独立trace，最终结果和晋级另行核验。不得将本目录旧binary微基准当作新正式性能。

仅通过基础功能检查，未完成模型精度验收。
