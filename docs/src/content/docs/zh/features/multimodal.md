---
title: "多模态支持"
sidebar:
  order: 60
---
本文档主要介绍xLLM推理引擎中多模态的支持进展，包括支持模型及模态类型，以及离在线接口等。

## 支持模型
- Qwen2.5-VL: 包括7B/32B/72B。
- Qwen3-VL: 包括2B/4B/8B/32B。
- Qwen3-VL-MoE: 包括A3B/A22B。
- MiniCPM-V-2_6: 7B。

## 模态类型
- 图片: 支持单图、多图的输入，以及图片+Prompt组合、纯文本Promot等输入方式。

## 服务级预处理参数

使用 `--mm_args` 传入按模态分组的 JSON 对象，参数交给对应的原生 processor 解释：

```bash
--mm_args='{"image":{"min_pixels":65536,"max_pixels":1048576}}'
```

`image`、`video`、`audio` 的参数相互隔离。当前接入以下图片参数；具体支持范围取决于模型使用的
processor，而不是框架中的模型名称白名单：

| 原生图片 processor | 参数 | 含义 |
| --- | --- | --- |
| Qwen2VLImageProcessor | `min_pixels`、`max_pixels`；或 `size.shortest_edge`、`size.longest_edge` | 图片面积预算，覆盖模型默认值；同一字段同时使用两种命名时，以 min/max_pixels 为准 |
| Glm4VImageProcessor | 同上 | 沿用 GLM 算法的时间维面积预算语义 |
| MiniCPMVImageProcessor | `max_slice_nums`、`scale_resolution` | 最大切片数和切片目标分辨率，同时应用于图片处理和 prompt 占位符 |
| KimiK25ImageProcessor | `in_patch_limit`、`patch_limit_on_one_side` | 输入 patch 预算和单边 patch 限制 |
| Mistral3ImageProcessor | `longest_edge` | 图片最长边限制 |

例如，Qwen 系列只覆盖最大面积时使用 `--mm_args='{"image":{"max_pixels":1048576}}'`，
MiniCPM 使用 `--mm_args='{"image":{"max_slice_nums":4,"scale_resolution":448}}'`。
未指定的字段继承模型配置与 processor 默认值。未设置、空字符串、`{}` 或空的模态对象均保持原配置。
参数在启动时解析并初始化，对服务期间的所有请求一致生效。

框架校验 JSON 结构和模态名称；字段类型、取值和组合由实际 processor 校验。
未被消费的字段在启动时报错，包括尚未接入参数接口的 video/audio processor，避免参数被静默忽略。
像素预算路径要求正整数、合并后的 `min_pixels <= max_pixels`，并至少能容纳一个对齐网格。
采用原生 token-budget resize 的配置暂不接受像素预算覆盖。

图片尺寸按模型的 patch/merge 规则对齐，`max_pixels` 不保证严格的视觉 token 上限。
极窄图片在紧预算下仍需保留非零的对齐尺寸，最终面积可能超过 `max_pixels`。
视觉 token 数由最终 `image_grid_thw` 计算。服务内多张图片分别应用相同参数。

CLI 中的 `--mm_args` 优先于服务配置文件。使用 `--config_json_file` 时，对应字段为 JSON 字符串：

```json
{"mm_args":"{\"image\":{\"max_pixels\":1048576}}"}
```

图片预处理统一使用 xLLM 原生实现，包括 Python 模型执行模式。
原有 `--image_max_tokens_num` 已移除，请改用 `--mm_args` 中的像素参数。
本入口不提供请求级覆盖。仅配置 `image` 时不会改变视频或音频行为。

### 接入新的原生 processor

复用现有 processor 的新模型自动获得相同参数支持。实现新的 processor 时，在构造函数中接收
`const ProcessorConfig&`，通过 `config.value<T>("image.参数名", 模型默认值)` 读取参数并校验算法约束。
装配流程创建 processor 时传入完整的服务覆盖参数，各 processor 自行读取需要的模态，构造完成后检查未消费字段；
无需修改 gflag、公共字段列表或模型分支。
旧的单参数构造函数仍支持空配置，提供非空但未消费的参数会明确报错。

`ProcessorConfig` 只在初始化期间使用，processor 保存解析后的数值。新参数如果也影响 prompt
占位符，应让相应的 prompt processor 读取相同字段，保持预处理与 token 展开一致。
例如 MiniCPM 的图片和 prompt processor 都读取 `image.max_slice_nums` 与 `image.scale_resolution`。
配置对象只提供参数，`create_processor` 负责创建处理器。

:::caution[注意事项]
- 目前多模态后端不支持prefix cache以及chunk prefill，正在支持中。
- 目前，xLLM统一基于JinJa渲染ChatTemplate，部署MiniCPM-V-2_6，模型目录需提供ChatTemplate文件。
- 图片支持Base64输入以及图片Url。
- 目前多模态模型主要支持了图片模态，视频、音频等模态正在推进中。

:::

## 多模态输入与缓存监控

以下三组指标为进程累计计数，进程重启后归零。标签只使用固定枚举，不包含 URL、请求 ID 或内容 hash。

| 指标族 | 维度 | 统计口径 |
| --- | --- | --- |
| `mm_input_stage_failures_total` | `stage=transfer/processor` | 请求实际进入 transfer 或 processor 阶段，普通返回失败时加一；成功返回会建立零值序列但不增加失败数。纯文本、未执行阶段和直接提交 `prompt + MMData` 的旁路不虚构样本。 |
| `mm_input_items_total` | `modality=image/video/audio/unknown`，`representation=raw/embedding/unknown`，`result=success/failed` | 每次经 `MMInputTransfer` 实际执行的单项媒体处理调用普通返回后记录一次；绕过该入口直接调用 handler 不计。成功和失败均按原始 `content.type` 精确匹配 modality 与 representation；两者描述提交的输入项，不从 `MMInputItem` 的解码结果或 embedding tensor 推断。VIDEO 与 AUDIO 共存时，音频是从同一视频文件解码出的音轨，因此只计一个 video；独立提交的音频项仍单独计数。加载、解码或预嵌入解析失败只改变 result，不改变输入分类，映射见下表。加载或解码失败只计一次；未启动任务不计。 |
| `mm_cache_lookups_total` | `cache=processor/encoder`，`outcome=hit/miss` | 每次真实 cache lookup 返回后记录一次。缓存关闭、无 hash、预嵌入或已 embedded 等未执行 lookup 的路径不计 miss。 |

成功和失败共用以下输入分类：

| 原始 `content.type` | `modality` | `representation` |
| --- | --- | --- |
| `image_url` | `image` | `raw` |
| `video_url` | `video` | `raw` |
| `audio_url` | `audio` | `raw` |
| `image_embedding` | `image` | `embedding` |
| `video_embedding` | `video` | `embedding` |
| `audio_embedding` | `audio` | `embedding` |
| 其他未识别或不支持的类型 | `unknown` | `unknown` |

representation 表示提交的输入形式，不代表已经解码或解析成功。处理结果只决定 `result=success/failed`；不按前缀、后缀、子串或大小写猜测类型。

当前 bvar 导出器将维度拼入指标名，而不是输出原生 Prometheus labels。例如：

```text
mm_input_stage_failures_total_stage_transfer
mm_input_stage_failures_total_stage_processor
mm_input_items_total_modality_image_representation_raw_result_success
mm_input_items_total_modality_image_representation_raw_result_failed
mm_input_items_total_modality_image_representation_embedding_result_failed
mm_input_items_total_modality_unknown_representation_unknown_result_failed
mm_cache_lookups_total_cache_processor_outcome_miss
mm_cache_lookups_total_cache_encoder_outcome_hit
```

输入分类调整不改变指标名和维度集合，但已知类型的失败不再归入 unknown：modality 按输入模态分类，representation 按提交形式分为 raw/embedding。失败总量查询应覆盖所有 modality 和 representation 的 failed；历史 unknown 序列不能直接拆成新分布，也不能跨新旧口径计算分类失败率。新口径下可按相同 modality/representation 计算 `failed / (success + failed)` 的同窗口增量；分子、分母使用相同进程集合，先分别求和再相除，分母为零时显示 N/A。

当前导出器对这些 MultiDimension 指标使用 gauge 类型声明，但上述指标的业务语义是累计 Counter；计算增量时需处理进程重启归零。

阶段失败率复用已有阶段样本数，不另增 success/total 指标：

- transfer 分母：`mm_transfer_latency_milliseconds_count`。
- processor 分母：`mm_processor_latency_milliseconds_count`。
- 失败率 = 同窗口失败计数增量 / 对应阶段样本数增量。分子、分母必须使用相同进程集合；跨实例先分别求和再相除。
- 分母只覆盖实际进入阶段且普通返回的请求，包括成功和失败。异常退出不在此口径内；语音入口提前处理也不由这些公共阶段指标补计。

缓存命中率分别按 processor、encoder 的 `hit / (hit + miss)` 增量计算，不混合缓存层。未出现的结果序列可按零处理，但没有实际 lookup 或窗口分母为零时应显示 N/A，而不是零命中率。worker 汇总代表实际 lookup 工作量，不是去重请求数。processor cache 命中不代表跳过下载/解码，encoder cache 命中也不代表省去了输入 H2D。

已有 `encoder_cache_hit_rate` 保持兼容；不要从其窗口平均值反推累计命中次数。已有 latency 指标是 bvar LatencyRecorder，不是标准 Prometheus bucket histogram。
