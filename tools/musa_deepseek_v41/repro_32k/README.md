# DeepSeek V4.1 Flash 32k/1k 双机复现

[2026-11-28 双场景阶段结果与接力入口](HANDOFF_20261128.md)包含 4.3.8 环境、A/B 各自的最新角色配置、NFS 权重路径和继续验证所需原件。下文保留 4.3.10 历史病例。

三个源码快照均从 `d7d14b1d19ee8e995aa8bd0f6a15e9a1a4fd1697` 分别恢复，并按原构建清单逐文件校验。

| 病例 | P / D | 原实测 |
| --- | --- | --- |
| G32PEXPM256 | G32PM2561（M256 分段专家 GEMM）/ G32DRING1 | 32k/1k C1，1 暖 2 测，TTFT 1749.03 ms，TPOT 11.36045 ms/token |
| G32DRINGC1 | G32RAGP1（支持非整齐的输入尾块）/ G32DRING1 | 32k/1k，按 2.8 请求/秒逐级测 C16/C24/C48。C48 的 48 条平均 TTFT 214.06 s、TPOT 29.65935 ms/token |

C48 含排队时间。48 条全部输出 1000 token，其中 34 条输入恰好 32000 token。两组是各自配置的短测，本提交做了离线校验。指南的 480 条长测及更换底座后的 GPU 复测待另行安排。

## 环境和所需资产

- SJ9 Prefill 8 卡 / SJ10 Decode 8 卡，宿主均为 **4.3.10**。每端将本机核验过的 4.3.10 用户态库目录传给 `--driver`，目录包含 `libmusa.so.1`。
- 运行底座是 `registry.mthreads.com/presale/inference/sglang:v0.5.20-dsv4.1flash-amd64-ph1-musa4.3.8-beta`，仓库 digest 为 `sha256:df97856006bf521c7174dfb007d3ef7e80398570214a0cf2ac66a9b70be92669`。镜像提供用户态环境，挂载后运行 xLLM C++ 服务。
- 运行容器将原实验资产根挂到 `/expt`，历史对应 `/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116`。需要原模型 view 及它引用的 **W4AFP8-mt1017** 权重和侧车。使用其中已验的 Engram、Indexer AOT、DeepEP 与 MATE 0.2.8 资产。各角色的路径和选路完整记录在 [roles.json](roles.json)。
- 将原部署服务包挂到 `/opt/xllm_service`，包括 `bin/etcd`、`bin/xllm_master_serving` 与依赖库。D 端需提供归档的 Router tokenizer 预初始化库，位于实验资产下 `long_opt_20261119/goal20k200k_20261119/diagnostics/ROUTERTKNINIT2/librouter_tokenizer_preinit.so`。M256 P 端需要 `G32RAGLIB1` 对应的 launcher、mubin 和清单。
- 镜像、权重、二进制、共享库和 ShareGPT 原始文本使用内部归档。仓库保存重建代码、配置与校验信息。端口和网卡按归档配置使用，脚本支持按现场传入 `--mccl-ifname` 和 `--rdma-devices`。同一套服务端口一次只跑一个病例。

## 1. 编译

构建使用 **PyTorch 2.9.0 / torch_musa**，由 C++ 链接 PyTorch 包内的 libtorch。三份编译产物来自原 CPU-only builder 的 CMake/Ninja 图。构建脚本恢复候选源码并校验文件哈希，再在无 GPU 的独立容器里执行已归档的 worker。两种 builder 的 Docker image ID、Ninja 图和原 ELF 哈希见 [build_manifest.json](build_manifest.json)。原 builder 默认保留在原构建机，本地缺少时需要从内部环境取得同 ID 镜像。运行底座单独拉取后缺少该构建图。

在包含基线 Git 历史的本次检出目录运行。以下目录都要求是新的路径，父目录须存在。只需编译本次选定的 P 和共用 D；也可直接挂载原归档 ELF 并核对清单。

```bash
pkg=tools/musa_deepseek_v41/repro_32k
# 只恢复并校验源码
python3 "$pkg/build.py" G32PM2561 /tmp/v41-m256-sources
# 指定原实验资产根，启动实际编译
python3 "$pkg/build.py" G32PM2561 /tmp/v41-m256-build --assets "$asset_root"
python3 "$pkg/build.py" G32RAGP1 /tmp/v41-ragged-build --assets "$asset_root"
python3 "$pkg/build.py" G32DRING1 /tmp/v41-decode-build --assets "$asset_root"
# M256 launcher 与 mubin 也可单独重建
python3 "$pkg/build.py" G32RAGLIB1 /tmp/v41-m256-provider --assets "$asset_root"
```

ELF 在输出目录的 `bin/xllm`，`artifact.json` 保存哈希和动态依赖。启动时默认核对原实测 ELF 的 SHA-256。使用重编 ELF 时核对编译产物后传入 `--binary-sha256`。

## 2. 启动 1P1D

在已经挂载上述资产、各分配八卡且端口空闲的两个容器内执行。两端使用相同的病例、同一共享新运行目录和同一新 namespace。先启动 SJ10 的 D 命令，它负责启动 etcd、Router 与 Coordinator。再在 SJ9 启动 P。以下变量均为容器内路径或地址。C1 设 `case_id=G32PEXPM256`，P 用 M256 ELF 并提供 launcher。C48 设 `case_id=G32DRINGC1`，P 换用 ragged ELF，并省略 `--ragged-provider`。

```bash
# SJ10
python3 "$pkg/run_role.py" "$case_id" d --run "$repro_run" --namespace "$repro_namespace" \
  --p-ip "$prefill_ip" --d-ip "$decode_ip" --model "$model_view" \
  --binary "$decode_elf" --driver "$decode_libmusa_dir" --router-preload "$router_preload"

# SJ9
python3 "$pkg/run_role.py" "$case_id" p --run "$repro_run" --namespace "$repro_namespace" \
  --p-ip "$prefill_ip" --d-ip "$decode_ip" --model "$model_view" \
  --binary "$prefill_elf" --driver "$prefill_libmusa_dir" --ragged-provider "$m256_provider"
```

两端各写出 `prepared_config.json`，可核查实际的八 rank 命令和环境。结束后向两端前台监督进程发 SIGTERM，等所属子进程退出，分别查看两台机的进程与显存释放。

## 3. 测速与复算

用原归档的前 48 条输入，或从同一份 `ShareGPT_V3_unfiltered_cleaned_split.json` 与 tokenizer 重建。准备脚本依赖 Python 的 `tokenizers` 包，并逐条核对实测输入哈希。

```bash
python3 "$pkg/prepare_inputs.py" "$sharegpt_json" "$model_view/tokenizer.json" "$repro_inputs"
python3 "$pkg/benchmark.py" "$case_id" --endpoint "http://$decode_ip:31180" \
  --run "$repro_run" --inputs "$repro_inputs" --output "$repro_results"
```

测速脚本等待 P、D 和 Router 就绪。C1 运行 1 暖 2 测，C48 病例按归档间隔依次运行 C16/C24/C48。两种都以原始文本发流式请求，按请求 ID 桥接两端原生日志，核对首正文、唯一终态、实际输入输出长度及零 prefix 命中。结果保留 SSE、逐请求审计和 `summary.json`。同一 M256 部署可用 `--output-tokens 1` 复核 32k/1 快筛。
