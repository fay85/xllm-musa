# StageI提交前代码复核

范围：相对 d672197e5901a6a713941b4e751d535de247ff67 的本次优化；按仓库 AGENTS.md、code-review skill 和 custom-code-style.md 核对。没有复审或扩展无关历史改动。

## 优点

- `paged_attention.h:147` 的单行分支避开重复slot排序，保留多行stable latest-wins及先Attention读取再cache提交；不改变模型结构。
- `deepseek_v4_dsa_cache_ops.cu:2733` 严格检查shape/dtype/device/连续性/16B对齐，DeviceGuard + native MUSA当前stream；无CPU标量拷贝与异常后隐式切换。
- 双SJ各60项覆盖非默认stream上的producer/consumer、负/越界slot、空/环回prefill、decode与独立cache参考。正式两组各1+5，关闭profiler，同配置与30项未变依赖有独立核验。
- `convert_native_cpp_trace.py` 使用真实Record function id/名称/线程；scope异步GPU通过runtime correlation加入，不截短GPU区间或伪造frame时间。

## 问题

Critical：本次目标路径未发现必须修正的问题。

Important：本次实际V4.1 BF16/D512连续对齐路径未发现阻断提交的问题。2048输出不完全重复和历史默认配置长组超时是已记录的验收边界，不由本次bitwise cache单测证明已解决，不能对外称全模型精度达标。

Minor：`deepseek_v4_dsa_cache_ops.cu:2743` 新wrapper只支持明确的BF16/连续/对齐合同。未来若要让PagedAttention支持其他dtype、非连续输入或非8倍数维度，应在调用前按能力明确选路并加相应回归；不能把此API当通用scatter。当前模型D512实际输入满足合同。

## 建议与结论

后续优化先隔离长输入确定性和去除progress后稳定性；暂不把改变bounded replay/DeepEP等执行策略混入本次小优化。官方工具输出与trace开销分开报告。

Ready to commit：Yes，限本报告注明的C1/eager/TP8-EP8目标合同。4个C++文件clang-format20.1.6检查、4个Python文件Black25.1.0/py_compile、git diff --check通过；实际数值/正式性能/trace验证均有原始凭据。遵照用户要求不创建MR/PR。
