# T14：显式选择原生N-wide query块

源码核对还发现，非NPU平台原有configure_deepseek_v4_dspark_args会设置dspark_use_native_sas=false，DFlash/DSpark worker据此把一条请求展开为N个q1请求。我们已实现并通过56例组件检查的原生drafter接收的是一条请求、N个非因果query，而非这种展开形式。若保持原标志，会在NativeDraft的单请求检查处失败。

仅在显式XLLM_DSV41_DSPARK_NATIVE开启、MUSA V4.1专用drafter分支，设置dspark_use_native_sas=true，选择与实际实现对应的NATIVE几何。不启用NPU算子、不改变Python/V4分支、不删除边界检查。N<=5，target verify仍N+1<=6，当前不与Graph或多请求混用；CED与完整prompt aux capture冲突的组合提前拒绝。

此候选仍待原生C++完整服务检查，不因代码可编译就称为MTP已跑通或提速。仅通过基础功能检查，未完成模型精度验收。
