<!-- Copyright 2026 The xLLM Authors. All Rights Reserved. -->
# 发布验证记录（2026-10-06）

## 原实测组合

- Prefill：94/wjm_xllm，rank0 PID1605878，启动于2026-10-06 16:20:49；
  `/proc/1605878/exe` SHA256 为 `caf3db8f775224045f9b8143f1b8a3491fb4637229fd5ab18194e00b60506b43`。
- Decode：90/wjm_xllm，rank0 PID1014380，启动于2026-10-06 09:41:47；
  `/proc/1014380/exe` SHA256 为 `5ea86017debbd94ba751a8ce38c74c1e2e9d39487b4ef3fcf662a24cedbaac93`。
- 上述 PID/启动时间及 binary 在本次 GSM8K 完成后再次核验，未更换 P/D。
- GSM8K：完整1319题，原版评分1275正确，0请求错误，2截断。
- scoring_input SHA256：`0cea54254eee6da26475e5c76332d488fd35045bfbfa6c6a1940d4028654e683`。
- scored JSONL SHA256：`11a9513bc4db95688b0a66d0bd6d9d95ff1608a167d7c8e35327122f3e145c92`。
- summary SHA256：`4376ea7f94bc55180e05b8d0ec220474fb1d9a2a3243c99802f38f5fa28f2285`。
- 30K 性能来源为 `paired_audit.json`；本次发布过程没有重跑/覆盖之前配对结果。

## 发布源码与工具

- P/D 源码从冻结构建输入收集；未把活动 6× 优化工作区作为发布源。
- 28/xllm_musa_build_0914：P 12个、D 5个相关翻译单元，独立增量编译、archive替换及链接通过。
- P 构建目录 `/tmp/pd_release_prefill_build_20261006_r3`。
- D 构建目录 `/tmp/pd_release_decode_build_20261006_r2`。
- 先前构建/整理失败记录保留，最终仅引用上述成功结果。
- 90/wjm_xllm：全部发布 shell 脚本 `bash -n` 通过，Python `compileall` 通过。
- `deepseek_v4_stream_benchmark.py --help`、`run_dspark_stress.py --help` 通过。
- 发布版 GSM8K runner 对完整1319题执行 dry-run：冻结协议/hash 校验通过，发送请求0次。
- 压测 runner SHA256 与原配对记录一致：
  `0e6d4ac564292ca4e70a715f82e198ec31a24a316fde0bec9cbc056f80bee375`。
- Git staged whitespace 检查通过，新增文本为LF；保留实际源码内容与编译输入校验。

## 明确未做的验收

新重编 binary 尚未替换服务做完整精度/性能复测；空镜像的无缓存全量 CMake 构建未验证。
本次未重新运行 SGLang 全量推理，也未把历史2.5K/1.5K成绩标为本轮重测。
当前30K为严格递增计数，不声明所有自然语言长上下文任务均通过。

无 GPU reset；未停其他任务、未改模型/评分标准，原运行服务保留。
