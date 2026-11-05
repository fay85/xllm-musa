# 本轮改动自查（提交前，非MR）

范围：`d2f5bda9`之后的当前工作区改动及新增MegaMoE模块/回归脚本。2026-11-05最终提交前更新；只提交既有个人分支，不创建MR/PR。

## 已有可靠依据

- `xllm/python/models/glm5_next.py:2435`把MATE返回的完整路由结果与另做TP归约的共享专家相加；CPU控制契约和真实GPU算式对照共同覆盖了重复/遗漏归约风险。
- `xllm/python/models/glm5_next_weight.py:89`保留原始FP8字节与显式block scale，拒绝错误维度、dtype、缺失或非法scale；真实checkpoint第3层回归已通过。
- `xllm/python/kernels_musa/mega_moe.py:158`明确选择支持clamp的split-prefill，并清除无效padding路由；不以异常捕获来改走另一个计算实现。
- `xllm/python/models/glm5_next.py:3281`在复制checkpoint前恢复mHC控制参数FP32，避免先舍入后提升类型；v6隔离实测和v12固定12题完成，当前v14c真实1P1D的0062为D、301Token。

## Critical：当前固定拓扑下未发现新增的确定性代码错误

这不是全模型数值验收结论。已观察到同前缀full-prefill与逐token decode的关键数字不一致，原因仍在验证；不能据此宣布所有模型数值语义已验收通过。用户最新明确要求记录问题后测速，故后续数据定位为保留精度边界的性能基线。

## Important：验证/能力边界

1. `xllm/python/kernels_musa/mega_moe.py:164`把小输入补至129行并执行同步/barrier，开销显著，正式C1两组TPOT约323ms/token，尚无去除padding/barrier的配对性能归因。保留正确性方案，待得到MATE支持clamp且经回归的decode路径后再优化；不在本轮静默移除同步。
2. `xllm/python/kernels_musa/mega_moe.py:34`使用组大小和拓扑配置约束IPC路径，不能单凭EP<=8推断任意外部启动配置都在单机。当前部署脚本固定同一服务器8rank，跨机仅P/D状态传递；未承诺跨机EP通用支持。若以后开放外部任意EP启动，应增加物理主机一致性预检。
3. `xllm/python/kernels_musa/mega_moe.py:65`提供显式工作区close；独立回归结束调用，原生服务目前依赖既有进程生命周期。当前sj9多次正常TERM八rank退出0，并不等于长期热重载/资源回收全场景已通过。sj10旧BF16进程的MCCL退出等待是独立既有阻塞，不能掩盖。
4. v14c随机模型1P1D的4/4/140token三请求与单机文本/计数/finish完全一致，双方16rank正常退出；真实模型三问正确，0062/0065当前正常结束为D/A。完整100题已完成：本地92/100、官方93/100、一致率95%；2048输入核验码检查通过（实际32输出，预算256）。graph/CP/DP等准入仍拒绝；这些样本不等于所有长上下文边界均通过。
5. 新增recurrent KDA、FP32 TP归约均为显式开关，默认chunk/native不变；只支持已检查的MUSA/BF16配置，packed recurrent提前拒绝，无异常回退。CPU559通过116跳过、GPU KDA10通过，但逐token prefill与FP32通信有额外性能代价。
6. v13强制135token一致输入仍出现增量2/full5的关键预测反例，最终层relative L2=0.275983，logits=0.229722。固定题答对不能消除此问题；保留后续定位，正式结论不能写“全部数值机制已无误”。

## Minor

- 新增Python函数有类型声明；新增文件有2026版权头，正式库代码使用共享logger；源码、测试及撰写的Markdown通过`git diff --cached --check`。原始CSV的CRLF、终端日志和绘图库SVG保留原样，不为消除空白提示改写实验证据。
- 本轮不改C++，不需要为了Python变化重新编译native二进制。MATE的JIT编译产物单独记录。

## 建议与当前判断

适合以实验基线提交到既有个人分支，**不适合标为全部数值验收完成或通用生产支持**。
用户已明确要求保留问题、停止后续分歧复测/根因追查并进入性能，不抓trace；按此范围交付。
当前100题与2048输入检查完成，仍保留0076/0088/0098及严格前缀反例。
两组性能已完成，各1暖机+5正式；全部12条原始请求的Token数已与native完成日志交叉校验，最终结果已写入性能结果.md。
临时KDA参考、逐层dump、未采用的编译桥接实验不混入当前默认服务代码。
11:01核对，12个源码/测试文件仍逐一匹配实际部署的v14c哈希清单。
