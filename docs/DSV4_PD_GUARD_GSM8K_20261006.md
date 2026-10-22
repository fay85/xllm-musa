<!-- Copyright 2026 The xLLM Authors. All Rights Reserved. -->
# DSpark 1P1D Guard：GSM8K 全量精度复现

## 1. 测试对象与结果状态

本报告测试的是 **94P + 90D，经 92 Control 入口的 1P1D**，不是 96 单机服务。
P binary SHA=`caf3db8f775224045f9b8143f1b8a3491fb4637229fd5ab18194e00b60506b43`；
D binary SHA=`5ea86017debbd94ba751a8ce38c74c1e2e9d39487b4ef3fcf662a24cedbaac93`。
保持同一套 30K 发布配置：N6/targetq7，Graph on，P overlap off、D overlap on，Prefix on。

2026-10-06 发布前重新完成了 1319 题全量测试及原版 SGLang 离线评分：

| 指标 | 本次结果 |
|---|---:|
| 官方 symbolic 正确数 / 全量 | **1275 / 1319** |
| 官方准确率 | **96.66413950%** |
| HTTP / 请求错误 | **0** |
| stop / length | 1317 / 2 |
| 失败分类 | 答案不符40、截断错误2、未提取到 boxed 答案2 |
| 附加 numeric boxed exact match | 1254 / 1319 = 95.07202426% |

主成绩严格使用原版 MathEvaluator，不能把 supplementary numeric 判分混作主成绩。
两种判分有21题差异，完整 ID 列表、错题和 scorer hash 见
[本轮评分 JSON](evidence/pd_guard_20261006/pd_guard_full.summary.json)。
**这是当前 P/D 的新测结果，不是旧版或单机分数。**

测试目录：
`/data/nfs_shared/jiaming.wang/pd_v128_30k_20261005/release_gsm8k_20261006`。

SGLang 的历史同协议参考是 1276/1319 = 96.73995451%；本次少答对1题，差约0.0758个百分点。
该参考不是本次重新运行的 SGLang，也不意味着两边错题完全相同。
30K 计数 64/64 不等价于 GSM8K 全量通过；两种验收分别报告。

生成结果完整覆盖1319个唯一ID、0重试；评分输入 SHA：
`0cea54254eee6da26475e5c76332d488fd35045bfbfa6c6a1940d4028654e683`。
28 评分目录：`wjm_sg_dspark:/tmp/pd_guard_release_eval_20261006`；
完整 scored JSONL、score.log 和 hash 清单也归档回原测试目录的 `official_score/`。

## 2. 固定协议

| 项目 | 设置 |
|---|---|
| 数据 | GSM8K test 全量 1319 题，固定顺序 gsm8k-0～1318 |
| prompt | SGLang sgl-eval 单轮 boxed-answer prompt，原始 payload 不改写 |
| max_tokens | 1024 |
| temperature / top_p | 0 / 0.95 |
| chat_template_kwargs | `{"thinking": false}` |
| 客户端并发 | 1（不等于服务端容量1；服务容量仍16） |
| 失败重试 / 删题 | 0 / 不允许 |
| HTTP 模型别名 | `DeepSeek-V4-Flash-0731-FP8-mt` |
| 评分 | 原版 `sgl_eval._vendored.nemo_skills.evaluator.math.MathEvaluator(config={})` |

冻结请求原 alias 为 `DeepSeek-V4-Flash-FP8`，发 HTTP 时**只改模型别名**；
messages、采样参数和输出上限保持不变。模型别名只用于服务路由，实际权重是
`/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt`。

不是旧 `bench_other.py` 的 5-shot/256 协议。不能直接拿两个协议的百分比判断实现好坏。
生成脚本不读答案；离线 adapter 和 scorer 才读 reference。错误、空输出和截断仍计入分母。

固定资产根：
`/data/nfs_shared/jiaming.wang/pd_n6q7_r3_20260927/acceptance_assets_20260927`。

| 文件 | SHA256 |
|---|---|
| `gsm8k_frozen/requests.json` | `3cf14a90c1f64a3a2e76cd920adc13624278d824dc23001b3cb18302ad4020c0` |
| `gsm8k_frozen/dataset_test.jsonl` | `67e9046d913470477dfb321814d4e3240240a397cfc19c6c610af02dd2cb6456` |

这些 benchmark 资产不重复提交 Git。复制资产时保留请求、dataset、request_index.json 和哈希；
不能自行调整 prompt 后仍将它叫作同协议成绩。

## 3. 生成：在 90 / wjm_xllm 执行

先按 [启动手册](DSV4_PD_GUARD_1P1D_REPRO_20261006.md) 确认 P/D/Control 健康。
停止其他客户端负载，不停止当前 P/D 服务、不改配置，不与性能压测同时执行。

```bash
export REPO=/data/nfs_shared/jiaming.wang/xllm_pd_guard_release_20261006
export ASSETS=/data/nfs_shared/jiaming.wang/pd_n6q7_r3_20260927/acceptance_assets_20260927
export GSM_RUN=/data/nfs_shared/jiaming.wang/gsm_pd_release_$(date +%Y%m%d_%H%M%S)
cd "$REPO"
python3 -u tools/pd_guard_20261006/gsm_frozen_runner.py \
  --assets "$ASSETS" --output-dir "$GSM_RUN" --count 1319 --timeout 300
```

输出目录必须不存在。保存：

- `manifest.json`：协议、输入 SHA、每个实际 HTTP payload 的 SHA。
- `payload_N.json` / `http_payload_N.json`：原请求与实际发出的 alias 版本。
- `response_N.http.body` / `response_N.json`：未经清洗的原始回答。
- `transport.jsonl`：每题 HTTP 状态、耗时、错误和 body SHA。
- `complete.json`：仅代表生成过程完成，不代表答案正确。

`--count 10` 或 `--count 20` 仅供小样本诊断，不能作为全量准确率。
`--dry-run` 不发送请求，只校验/保存 payload；也不能作为服务通过证据。

## 4. 转换：离线整理为原评分器格式

```bash
python3 tools/pd_guard_20261006/gsm_to_sgl_rows.py \
  --run-dir "$GSM_RUN" --assets "$ASSETS" \
  --output "$GSM_RUN/scoring_input.jsonl"
sha256sum "$GSM_RUN/scoring_input.jsonl" "$ASSETS/gsm8k_frozen/dataset_test.jsonl"
```

adapter 要求完整1319条、无重复/重试，校验保存的响应 body SHA。
HTTP 失败仍产生错误行，并由评分器计为错误，不得排除后再算百分比。

## 5. 官方离线评分：28 / wjm_sg_dspark

该容器已经具备原 SGLang 评分依赖，**此步骤只用 CPU，不运行模型**。
已确认版本：`sgl-eval 0.0.1`、`math-verify 0.9.0`、
`latex2sympy2-extended 1.11.0`、`sympy 1.14.0`。

28 的共享盘与 90 的共享盘不同。将本分支、`scoring_input.jsonl` 和固定 dataset
通过 SCP（必要时经 Windows 中转）复制到 28，再 `docker cp` 到 `wjm_sg_dspark`。
例如在容器内整理成：

```text
/tmp/pd_guard_release_eval/
├── tools/eval_gsm8k_sgl_protocol.py
├── tools/score_gsm8k_sgl_reference.py
├── scripts/logger.py
├── inputs/scoring_input.jsonl
├── inputs/dataset_test.jsonl
└── logs/                           # 评分输出必须在此 checkout 的 logs 内
```

```bash
cd /tmp/pd_guard_release_eval
python3 tools/score_gsm8k_sgl_reference.py \
  --dataset inputs/dataset_test.jsonl --input inputs/scoring_input.jsonl \
  --output-dir logs/pd_guard_full --label pd_guard_full
```

正式评分不加 `--partial-diagnostic` 或 `--limit`。
读取 `logs/pd_guard_full/pd_guard_full.summary.json`：
`complete=true`、`examples=1319`、`unique_ids=1319`，再查看
`symbolic_correct`、`symbolic_accuracy_percent`、`request_errors`、`truncated`、
`failure_categories` 和 `failed_ids`。

主成绩用原 MathEvaluator symbolic score；`numeric_boxed_accuracy_percent` 是附加对照，
不擅自更换主判分器。保留 scorer 实现 hash、依赖版本、raw/scored JSONL。

## 6. 可比性和验收边界

- GSM8K 模型服务始终使用原实测 P/D binary，不使用仅完成编译检查的新候选。
- 本轮属于当前服务请求历史下的串行全量评测，不能假设所有题目冷 cache。
- 若比较 SGLang，应固定同一 `requests.json`、权重、采样参数、输出上限和 scorer，
  分别保留模型别名差异，不改题、不挑错题子集代替全量。
- 若准确率下降，先固定 SGLang 正确/xLLM 错误的同一题及相同 token IDs，比较首次 token/state
  分叉，再逐层定位；本发布工作不会为追求分数修改 prompt 或参考答案。
