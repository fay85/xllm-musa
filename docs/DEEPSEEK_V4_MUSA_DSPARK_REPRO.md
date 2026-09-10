# DeepSeek-V4-0731 MUSA DSpark 复现

## 已验证链路

- 工作目录：`/data/nfs_shared/jiaming.wang/xllmmerge/xllm0814/xllm-dsv4-musa-dspark`
- 分支：`dsv4/feat/musa-dspark-20260910`
- 算法代码版本：`a9cd485c`。后续文档提交不改变该二进制。
- 二进制 SHA256：`78f4a5acff37b5353510bdf1b4d3a20aa8cb455c6b2e683f584b10d322eceee5`
- 模型：`/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt`
- API 的 `model_id` 保留 `DeepSeek-V4-Flash-FP8` 作为接口标签，实际权重版本以以上 0731 路径和 manifest 为准。
- 8 卡 S5000，TP8/EP1/DP1/CP1，batch=1；DSpark 五个草稿 token。
- **target 六 token Graph + draft eager block**，不是整个 DSpark pipeline 都已 Graph 化。
- prefill SP；Prefix Cache 开启，输入共享率 0.65，实测命中 1536 tokens。
- schedule overlap 关闭；静态 greedy，adaptive/confidence 策略未启用。
- 输入/输出 2500/1500，严格 `1..750`；block size 128，max batch tokens 10240，memory utilization 0.8，response threads 4。

两个独立服务的完整热窗均为 variant 2–7：

| 服务 | 平均 TTFT / ms | 平均 TPOT / ms | 严格精度与缓存 |
| --- | ---: | ---: | --- |
| dspark_sp_prefix_07 | 217.370500 | 8.212000 | 6/6 全文 750/750 PASS，cached=1536 |
| dspark_sp_prefix_08_fresh | 215.821167 | 8.188667 | 6/6 全文 750/750 PASS，cached=1536 |

每个正式请求 usage=2500/1500，两窗均满足 TTFT<250 ms、TPOT<14 ms 且不高于参考门限 13.758500 ms。冷态/建缓存请求和全部失败实验另行保留，没有筛选热窗样本。

## 环境与构建

验证环境为宿主机 `10.20.34.28`、driver 5.2.0-server、MCC 5.1.0、Python 3.10、Torch/TorchMUSA 2.9.1。复用机器现有库，未下载 LibTorch/NVIDIA CUDA。

```bash
export DSPARK_ROOT=/data/nfs_shared/jiaming.wang/xllmmerge/xllm0814/xllm-dsv4-musa-dspark
export DSPARK_TOOLS=/data/nfs_shared/DeepSeek_V4_MUSA_merge_runs/dspark_20260910
cd "$DSPARK_ROOT"

# 已有 build cache 时只做增量构建。
MUSAMAPPING_PATH=/usr/local/musa/tools/musamapping MUSA_HOME=/usr/local/musa \
  cmake --build build/dspark-musa-local --target xllm -j16
sha256sum build/dspark-musa-local/xllm/xllm
```

若需要重新配置，先检查并使用 `$DSPARK_TOOLS/configure.sh`。它包含本机 Torch、MKL、ASIO、xxHash 和已有 vcpkg 路径。必须保留 `VCPKG_MANIFEST_MODE=OFF`、`VCPKG_MANIFEST_INSTALL=OFF`，不要对共享依赖树执行安装/替换。MUSA standalone 的 `USE_MUSA=ON`、`USE_CUDA=OFF` 保持不变；树外 toolkit-probe shim 只用于 CMake 探测，实际编译仍为 MCC/MusaMapping。

当前单体 target 的链接图仍包含 Mooncake，不能宣称已经完全跳过。构建失败、共享依赖恢复及检查记录见 [验证记录](DEEPSEEK_V4_MUSA_DSPARK_VALIDATION.md)。

还需要原性能链的本地 AOT 算子和 custom all-reduce：

- AOT 根目录：`/data/nfs_shared/jiaming.wang/test/workspace_fp8_28/cached_ops`
- custom-AR：`/data/nfs_shared/DeepSeek_V4_MUSA_merge_runs/xllm0814_perf_goal_20260817/custom_ar_current/libxllm_custom_ar_sglang_current.so`
- 底层启动器：`/data/nfs_shared/DeepSeek_V4_MUSA_merge_runs/xllm0814_perf_goal_20260817/run_external_tp8_target.sh`

上述外置依赖不属于 Git 源码。本文命令针对该共享 NFS 环境；其他机器必须先准备匹配的驱动、TorchMUSA ABI、模型和外置资产，不能仅拉取分支就宣称环境完备。

实际加载库清单见 `$DSPARK_TOOLS/dspark_sp_prefix_08_fresh/loaded_shared_libraries.txt`，SHA256 见 `$DSPARK_TOOLS/dependency_sha256.txt`。其中还包括 DSA Model1、local-prefill、MLA、TF32 HC prenorm 模块；不要把启动器的最小 preflight 列表误当作完整环境清单。

## 启动

先从宿主机确认全部 GPU 占用。只停止自己已核实的占卡进程，不停止其他任务，不 reset GPU。

```bash
ssh -p22 mccxadmin@10.20.34.28 mthreads-gmi
export DSPARK_RUN=$(mktemp -d "$DSPARK_TOOLS/repro_dspark.XXXXXX")
echo "$DSPARK_RUN"  # 在第二个终端使用这个实际目录

# 前台运行；本终端 Ctrl-C 由启动器清理其自己的八个 worker。
XLLM2_EXTERNAL_TP8_GRAPH=true \
XLLM2_EXTERNAL_TP8_PREFIX_CACHE=true \
  bash "$DSPARK_TOOLS/launch_eager.sh" "$DSPARK_RUN"
```

`launch_eager.sh` 是最初诊断脚本的历史名称。上面显式覆盖 Graph/Prefix 开关后，使用的是已验证的 target Graph 链路，HTTP 端口 18861、master 端口 20861。它固定选择当前 DSpark 工作目录二进制、0731 模型、N=5 和 overlap=false，并在启动前校验依赖、保存 `launch_manifest.txt`。

## 请求与判定

第二个终端设置相同的 `DSPARK_TOOLS` 和实际 `DSPARK_RUN`，等待 `/health` healthy 后：

```bash
set -euo pipefail
curl -fsS http://127.0.0.1:18861/health
for warmup in 1 2; do
  timeout 180s python3 "$DSPARK_TOOLS/benchmark_0731.py" \
    --url http://127.0.0.1:18861/v1/chat/completions \
    --prompt-tokens 64 --output-tokens 64 --count-to 32 \
    --output "$DSPARK_RUN/warmup_${warmup}.json" || exit $?
  python3 "$DSPARK_TOOLS/assert_counting_result.py" \
    "$DSPARK_RUN/warmup_${warmup}.json" || exit $?
done

# 两次短测都通过后再执行。不可与其他正式请求并发。
bash "$DSPARK_TOOLS/run_prefix_window.sh" "$DSPARK_RUN" 0 7 \
  | tee "$DSPARK_RUN/window_0_7.log"
python3 "$DSPARK_TOOLS/summarize_prefix_window.py" "$DSPARK_RUN" 2 7 \
  > "$DSPARK_RUN/window_2_7.summary.json"
```

variant 0/1 是冷态/建前缀过程，**不能挑选其中的快样本拼入热窗**。正式统计保留 variant 2–7 全部六个请求，要求：

- 每个请求服务端 usage=2500/1500、cached_tokens=1536。
- 全文只含按顺序的 `1..750` 和空白；不以 HTTP chunk 数代替 token 数。
- 平均 TTFT <250 ms；平均 TPOT <14 ms 且不高于参考门限 13.758500 ms。

测试脚本在原计时结束后额外保存全文和严格文本检查，不改变原计时点。首次处理请求的不同线程可能出现额外初始化开销；因此必须保留所有 warmup 和冷态记录，不能以短请求的 TTFT/TPOT 替代正式长请求。

独立复验须停止该服务、用宿主机确认八卡零显存/无残留、创建新日志目录、重新启动并重复完整流程。窗口前、中、后保存宿主机 mthreads-gmi。发现外部 GPU 任务、精度失败或 usage 缺失，该窗口不能作为有效性能结果。

## 验证边界

- 已验证范围仅为以上 checkpoint、TP8/EP1、batch 1、静态 DSpark N=5、2.5K/1.5K、0.65 前缀。
- 不等于 batch8/16、PD、CP、多机、adaptive 或任意前缀命中比例已验证。
- 其他模型和非 MUSA 设备未做运行回归，不以本次结果代替它们的验收。
- 0731 target-only **eager** 曾出现精度异常；target-only Graph 已通过 `750/750`。本成果不宣称 eager 问题已修复。
- draft block 仍为 eager；不宣称全 draft/context/Markov 路径已捕获为 Graph。
- 精度验收是严格计数协议，不等于 GSM8K 或任意任务全面精度对齐；沿用原性能分支的 compact Graph compressor 行为，其他长上下文任务需另行验证。
- 构建、每次失败原因、旧二进制、完整样本与最终结果均保留在验证记录及 NFS 实验目录。未推送远端、未 reset GPU。

请在独立 DSpark 目录 review/commit。原 `xllm-dsv4-musa-perf` 目录另有与工作文件不一致的暂存改动，本次未清理；在原分支提交前先检查 `git diff --cached`。

## 后续工作与已达标项的区别

上述两轮 TTFT/TPOT 和严格计数门槛已经完成，不需要把历史记录中的“待验证”阶段误当作当前状态。继续推进时应单独定义新验收项：

1. **更广泛的精度验证。** 当前 `freeze_compact_graph_compressor` 路径保留 prefill 建立的压缩前缀，SWA 继续更新。需验证/完善 C4/C128 状态演进、拒绝草稿 token 后的状态恢复，以及短 suffix 的高前缀命中场景；计数 PASS 不能证明这些情形正确。
2. **target-only eager 精度修复。** 这是已记录的独立失败路径，不能写成已修复。
3. **draft 全链路 Graph。** draft block、上下文注入和 Markov 采样尚非全链路 Graph。仅将更多操作捕获成 Graph，不能替代前两项精度工作。

这些是后续工作清单，不是本轮新增实验结果；已有两个完整热窗和对应二进制保持不变。
