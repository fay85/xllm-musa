# 两组性能准备方案（用户已延期，未实测）

> 最新决定：两组正式性能不再于本轮执行。下文为已准备方案和历史准备核查，不是实测结果。未来MegaMoE接入及正确性排查后再实施。

精度检查先于性能测试。真实模型基础问题和固定MathQA10题协议验证已通过；按当前范围，完整100题及不一致题复测检查完成后才进入性能测试。性能使用 EvalScope 官方发布包 1.14.1 的 perf 客户端，CPU 独立虚拟环境在共享盘 `eval/venv`；不向推理环境安装其依赖。

## 固定输入与参数

- 目标部署：sj9 Prefill TP8 + sj10 Decode TP8，1P1D；MCCL，eager，FP8 checkpoint 分片后反量化 BF16。
- Chat API：`http://10.121.32.9:49880/v1/chat/completions`。
- 输入两组：包含真实模型聊天模板的 256 / 2048 token，输出目标 256 token。每组 1 次独立预热 + 至少 5 次测量，并发 1。
- temperature=0，top_p=1，reasoning_effort=low；性能测试单独设置 ignore_eos=true，以检查固定输出长度。这项设置不用于 MathQA 精度评测。
- `eval/perf_inputs_v2/input256_output256.jsonl` 与 `input2048_output256.jsonl` 各 6 行完整请求体；模板 SHA 与每条输入 ID 在同目录保存。每组相同Prompt重复6次，prefix cache关闭。
- 不使用 Completions 的 token ID 数组：本服务的当前 V2 协议对此未跑通。采用 line_by_line 完整 Chat 请求 JSON，保留实际输入和参数。
- tokenizer 离线计算长度已检查；最终仍以服务端 usage 为准。若实际 prompt_tokens 或 completion_tokens 不符，应单独标记，不能把短输出伪称 256 输出 token。

## 指标和边界

保存 EvalScope 原始明细、日志、版本及命令，报告成功/失败数、TTFT、TPOT、端到端时延、请求吞吐、输出 token/s 和输入/输出 token 数。首次加载及算子 JIT 不混入稳态结果，预热独立保存。

Chat streaming 的 TTFT 可能包括 reasoning 输出的首个增量；是否包含在客户端统计中需按该版 EvalScope parser 和原始 SSE 验证。TPOT 通常按 (E2E-TTFT)/(实际输出 token 数-1) 计算；不能混淆单请求输出速度与多请求总吞吐。

这不是 H20/H200 对比、不是 graph 优化结果，也不是 ModelScope 榜单复现。本轮未完成实测前不填写任何性能数值。

## 初步冷/热差异观察（不作为正式性能提升结论）

MathQA 0002 与 0003 恰好都为120个输入token，题目内容不同。P端日志中的 TTFT 分别42.866秒和12.027秒，prefix cache 命中数都为0。这与前面可见的首次JIT现象一致，但不同题目的 MoE 路由也不同，不能把差值全部归因某个优化，更不能当作严格A/B性能提升。正式性能组仍需固定输入、独立预热和5次测量。证据为本地 `same_shape_prefill_warm_cache_evidence` receipt。

### 测速输入冻结为 v2（实测前调整）

每组改为同一个固定prompt重复6次：第1次预热，后5次正式测量。v2取已校验v1的第0行重复，保存到 `eval/perf_inputs_v2`；v1保留。这样同组内容、MoE路由和输出也尽可能一致，便于判断预热与重复测量的波动。prefix cache仍关闭，每次都完整Prefill。实际测试前后仍核对usage及是否发生新的JIT；若测量区间还有编译，应明确标记并在缓存热后重新完成5次。

## 执行前核查（15:08，未启动正式测速）

实际安装版本为EvalScope1.14.1。已检查该版本line_by_line插件：完整JSON对象原样传递，OpenAI插件对已存在的字段使用setdefault，不会用CLI默认值覆盖temperature/low/max_tokens/ignore_eos。流式响应优先读取服务端usage；如果缺少usage，不能仅接受客户端重新分词的长度来宣称固定256输出成功，应检查原始SSE。预检命令与插件源码SHA保存在证据目录的evalscope_perf_preflight及evalscope_perf_source_contract回执。

新入口49880使用启动时3600秒业务截止时间；为避免原49080入口运行时flags更新无效影响后续请求，实际脚本更新为NFS tools/run_perf_baseline_v4.py（新增完整100题与复测检查记录的执行前校验）。正式测量仍未开始。


## 为保留预热原件而调整记录方式（实测开始前）

检查EvalScope1.14.1发现其内置warmup分支直接跳过SQLite写入，无法保留预热的原始响应。因此改为每组先单独用官方EvalScope跑1次，目录标记warmup；随后另一次官方命令跑5次，目录标记measured。两个命令都设置内置warmup-num=0，避免多发一次隐含预热。

实际总数仍是每组1次预热+5次正式请求，输入内容不变，推理服务和GPU缓存不重启。只有measured目录的5条记录进入正式性能，warmup完整原件单独保留。未修改EvalScope源码或其计时逻辑。

待运行脚本为NFS `tools/run_perf_baseline_v5.py`，取代尚未执行的v3/v4准备稿。命令、阶段和输出路径全部保存到实验结果目录commands.json。

执行前最后增加了原始记录校验：每个阶段完成后从SQLite核对HTTP成功数、真实服务端usage、固定采样参数和计时公式，并导出可直接阅读的请求/响应JSON。预热若未达到目标token数，停止进入正式五次测量。最终待执行脚本为 `tools/run_perf_baseline_v6.py`，校验器 `tools/validate_perf_phase.py`；尚未启动实测。
