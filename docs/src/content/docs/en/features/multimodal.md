---
title: "Multimodal Support"
sidebar:
  order: 60
---
This document introduces the current multimodal support in the xLLM inference engine, including supported models, modality types, and offline and online interfaces.

## Supported Models
- Qwen2.5-VL: including 7B/32B/72B.
- Qwen3-VL: including 2B/4B/8B/32B.
- Qwen3-VL-MoE: including A3B/A22B.
- MiniCPM-V-2_6: 7B.

## Modality Types
- Images: supports single-image and multi-image inputs, image + prompt combinations, and text-only prompts.

## Service-Level Preprocessing Parameters

Pass a JSON object grouped by modality with `--mm_args`. Each native processor interprets its own parameters:

```bash
--mm_args='{"image":{"min_pixels":65536,"max_pixels":1048576}}'
```

The `image`, `video`, and `audio` objects are isolated. The following image parameters are currently
supported, depending on the selected processor rather than a framework list of model names:

| Native image processor | Parameters | Meaning |
| --- | --- | --- |
| Qwen2VLImageProcessor | `min_pixels`, `max_pixels`, or nested `size.shortest_edge`, `size.longest_edge` | Image area budgets; min/max_pixels take precedence over the corresponding size aliases |
| Glm4VImageProcessor | Same as above | Preserves the GLM algorithm's temporal area budget semantics |
| MiniCPMVImageProcessor | `max_slice_nums`, `scale_resolution` | Slice count and target resolution, shared with prompt placeholder generation |
| KimiK25ImageProcessor | `in_patch_limit`, `patch_limit_on_one_side` | Input patch budget and per-side patch limit |
| Mistral3ImageProcessor | `longest_edge` | Maximum image edge length |

For example, Qwen models can use `--mm_args='{"image":{"max_pixels":1048576}}'`, while MiniCPM can use
`--mm_args='{"image":{"max_slice_nums":4,"scale_resolution":448}}'`. Omitted fields inherit model settings
and processor defaults. An omitted flag, an empty string, `{}`, or an empty modality object preserves existing
settings. Parameters are resolved at startup and remain fixed for every request in the service.

The framework validates JSON structure and modality names. Each processor validates field types, values,
and combinations. Unconsumed parameters fail at startup, including parameters passed to video/audio
processors that have not adopted the configuration interface. Pixel budgets must be positive integers,
satisfy `min_pixels <= max_pixels` after merging, and fit at least one aligned grid cell.
Native token-budget resize configurations do not accept pixel-budget overrides yet.

Dimensions follow model patch/merge alignment, so `max_pixels` is not a strict visual token limit.
For very narrow images with tight budgets, keeping nonzero aligned dimensions can make the final area exceed `max_pixels`.
The actual token count follows `image_grid_thw`. The same settings apply independently to each image.

The CLI flag takes precedence over the service configuration file. In a file passed to
`--config_json_file`, use a JSON string:

```json
{"mm_args":"{\"image\":{\"max_pixels\":1048576}}"}
```

Image preprocessing uses xLLM's native implementation, including with Python model execution.
The previous `--image_max_tokens_num` flag has been removed; use pixel parameters in `--mm_args`.
This option does not provide request-level overrides. Setting only `image` does not change video or audio behavior.

### Adding a Native Processor

New models that reuse a processor automatically inherit its parameter support. A new processor can accept
`const ProcessorConfig&` in its constructor, read parameters with `config.value<T>("image.name", model_default)`,
and validate its algorithm's constraints. The assembly creates processors with the complete service overrides;
each processor reads the modalities it needs. Unconsumed fields are rejected after construction.
No new gflags, central field lists, or model-name branches
are needed. Existing one-argument constructors continue to work with empty configuration.

`ProcessorConfig` is used during initialization; processors retain resolved values. If a parameter also affects
prompt placeholders, the prompt processor should read the same fields to keep token expansion consistent.
For example, MiniCPM's image and prompt processors both read `image.max_slice_nums` and
`image.scale_resolution`. Configuration objects only provide parameters; `create_processor` handles construction.

:::caution[Notes]
- The multimodal backend does not currently support prefix cache or chunked prefill. Support is in progress.
- xLLM now renders ChatTemplate uniformly based on JinJa. When deploying MiniCPM-V-2_6, the model directory must provide a ChatTemplate file.
- Image inputs support Base64 data and image URLs.
- Multimodal models currently mainly support the image modality. Video, audio, and other modalities are in progress.

:::

## Multimodal Input and Cache Metrics

The following metric families are process-cumulative counters that reset on process restart. Dimensions use fixed enums, never URLs, request IDs, or content hashes.

| Metric family | Dimensions | Counting boundary |
| --- | --- | --- |
| `mm_input_stage_failures_total` | `stage=transfer/processor` | Increment once when an executed request stage returns an error. Successful returns create a zero-valued series without increasing failures. Text-only requests, unexecuted stages, and direct `prompt + MMData` submissions do not create synthetic samples. |
| `mm_input_items_total` | `modality=image/video/audio/unknown`, `representation=raw/embedding/unknown`, `result=success/failed` | Record once after each per-item media-processing call actually executed by `MMInputTransfer` returns. Direct handler calls that bypass this entry point do not count. Both successful and failed calls derive modality and representation from exact matches on the original `content.type`. These labels describe the submitted input item, not the decoded output or embedding tensor. When VIDEO and AUDIO coexist, the audio track was decoded from the same video file, so this counts as one video item; separately submitted audio items still count independently. Loading, decoding, or pre-embedded parsing failures change only result, not the input classification listed below. A loading or decoding failure counts only once. Unstarted tasks do not count. |
| `mm_cache_lookups_total` | `cache=processor/encoder`, `outcome=hit/miss` | Record only after an actual cache lookup. Disabled caches and paths that skip lookup, such as missing hashes, pre-embedded inputs, or already embedded items, do not count as misses. |

Successful and failed calls use the same input classification:

| Original `content.type` | `modality` | `representation` |
| --- | --- | --- |
| `image_url` | `image` | `raw` |
| `video_url` | `video` | `raw` |
| `audio_url` | `audio` | `raw` |
| `image_embedding` | `image` | `embedding` |
| `video_embedding` | `video` | `embedding` |
| `audio_embedding` | `audio` | `embedding` |
| Other unrecognized or unsupported types | `unknown` | `unknown` |

Representation describes the submitted input format, not whether decoding or parsing succeeded. The processing outcome determines only `result=success/failed`. Types are not inferred using prefixes, suffixes, substrings, or case-insensitive matches.

The current bvar exporter encodes dimensions in metric names rather than native Prometheus labels. Examples:

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

This input-classification change preserves the metric family name and dimension set, but known-type failures no longer use unknown: modality follows the input modality, and representation is raw/embedding according to the submitted format. Total-failure queries must include failed series across all modalities and representations. Historical unknown series cannot be split into the new distribution; do not calculate classified failure rates across the change. Under the new definition, compute `failed / (success + failed)` from deltas over the same window and process set for each modality/representation pair. Sum numerators and denominators separately before dividing; a zero denominator means N/A.

The current exporter declares these MultiDimension metrics as gauges, but their application semantics are cumulative counters. Account for process resets when calculating deltas.

Stage failure rates reuse existing sample counts rather than adding success/total metrics:

- Transfer denominator: `mm_transfer_latency_milliseconds_count`.
- Processor denominator: `mm_processor_latency_milliseconds_count`.
- Failure rate = failure-counter delta / corresponding stage-count delta over the same window and process set. Sum numerators and denominators separately before dividing across instances.
- Denominators cover executed stages that return normally, including success and failure. Exception exits are outside this contract; earlier speech-service processing is not backfilled into these shared-stage metrics.

Compute processor and encoder cache hit rates separately using `hit / (hit + miss)` deltas. An absent outcome series can be treated as zero, but no actual lookups or a zero window denominator means N/A, not a zero hit rate. Worker aggregation measures lookup work, not deduplicated requests. A processor-cache hit does not imply that download/decoding was skipped, and an encoder-cache hit does not imply that input H2D was avoided.

The existing `encoder_cache_hit_rate` remains compatible; do not reconstruct cumulative hits from its window average. Existing latency metrics use bvar LatencyRecorder, not standard Prometheus bucket histograms.
