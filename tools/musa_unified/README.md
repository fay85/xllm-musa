# MUSA DeepSeek / Qwen：统一编译、复现与 xllm_pack

适用镜像：`registry.mthreads.com/presale/devtech/xllm:d1029-opt3`。
本次核对的image ID：`sha256:f19c55daf710fc2899b057a3faf62c82c1a72cf4f0c67faf420520efbd3a8407`。

一个源码树，两个角色产物。Prefill和Decode均包含DeepSeek-V4与Qwen实现，但编译时DeepSeek角色选择不同，因此不能用一个不明来源的`xllm`覆盖两者。每个二进制与自己的`libasio.so`配对；两模型的AOT也分目录保存。

本次工具/pack的实际验证范围见[验证记录](VALIDATION_20261031.md)：隔离目录Qwen68/68、120条性能门禁通过；DeepSeek仅完成Control CPU启动，完整PD/GSM8K仍待健康机器验收。

## 1. 当前已经验证的版本在哪里

共享盘现有构建：

```text
/data/nfs_shared/jd/d1029_compare_20261031/unified/
  source/
  build/prefill/               # 已有编译缓存
  build/decode/
  bundle/prefill/xllm
  bundle/prefill/libasio.so
  bundle/decode/xllm
  bundle/decode/libasio.so
  bundle/cached_ops/           # DeepSeek
  bundle/qwen_cached_ops/      # Qwen
  bundle/mate_mubin/
  build_unified_musa.sh        # 当时已执行的入口
```

现有产物P SHA256：`097585d0cbcdb9cf6349067a595e4d1faa2d33e3ef5e07dbb6d9f40c3de0203e`；D：`31677d967f016abc928d7df42ee7d29a1c1b04d2c8590766c60395b7471a80d9`。

该二进制组合在94上Qwen五图68/68、两轮120/120正文/usage/cache门禁通过；K2+Graph、C6、20K±1K文本加5图、100输出、缓存16384，热态TTFT **532.0 ms**、TPOT **15.22 ms**。这是此输入集的验收，不是通用图片准确率100%。

**DeepSeek新联合版的PD/GSM8K实测尚未完成。**90 GPU3重启后仍能在独立显存字节测试中复现错误；不要用它作为正确性验收机器。历史旧联合版20K/100、80%缓存、2P1D+PCP、DSpark K6+Graph为B1约488.85/9.228 ms，B16连续32请求约2023.02/17.199 ms；这些不能冒充新包成绩。

本目录把此前临时入口整理为仓库内工具。保留的生产改动包括Gate/compact MoE 2048行下限、grouped FP8 63行支持、VLM文本draft eager路由和完整五图输入构造；不包含NaN重试、layer dump或逐算子同步探针。

## 2. 编译命令

### 复用现有源码和缓存

先停止所有机器上使用目标bundle的服务，再在构建容器内执行；脚本会更新目标二进制/库，不能原地覆盖正在使用的包。

```bash
cd /你的Git仓库
R=/data/nfs_shared/jd/d1029_compare_20261031/unified
XLLM_SOURCE_DIR="$R/source" \
XLLM_BUILD_ROOT="$R/build" \
XLLM_BUNDLE_DIR="$R/bundle" \
MAX_JOBS=12 NINJA_TARGET=xllm bash tools/musa_unified/build.sh
```

### 编译当前Git checkout

在独立checkout中使用镜像已有依赖。不要覆盖已经初始化、可能含用户修改的submodule：

```bash
# 仅在third_party子模块目录为空时运行；实际镜像依赖版本会单独留档。
python3 tools/musa_unified/seed_image_deps.py \
  --image-source /workspace/xllm-mt-main \
  --report /data/build_evidence/image_dependencies.json

# d1029-opt3原镜像缺少默认Rust/Cargo工具链。可复用以下已存在的共享盘依赖，
# 或显式提供等价的RUSTC、CARGO、CARGO_HOME；不会自动联网安装。
export RUSTC=/data/nfs_shared/jd/dspark_merge_20261016_1050/tooling/rust/stable-toolchain/bin/rustc
export CARGO=/data/nfs_shared/jd/dspark_merge_20261016_1050/tooling/rust/stable-toolchain/bin/cargo
export CARGO_HOME=/data/nfs_shared/jd/dspark_merge_20261016_1050/deps/cargo-home
export XLLM_DEEPSEEK_OPS=/data/nfs_shared/jd/mt_feihu_build_20261029/hc027_r14/ops
MAX_JOBS=12 NINJA_TARGET=xllm bash tools/musa_unified/build.sh
```

默认输出`build/musa_unified/bundle/`，P/D缓存分别为`build/musa_unified/prefill/`和`decode/`。可用`XLLM_BUILD_ROOT`、`XLLM_BUNDLE_DIR`改到空间充足的盘。默认增量构建，只有显式`FORCE_CMAKE=1`才重新配置；不清空vcpkg，不FULL_RESET。源码/角色与已有cache不符会拒绝继续。

构建采用`USE_MUSA=ON`、`USE_CUDA=OFF`、`musamapping/mcc_wrapper`，不需要nvcc或MTTOplib。`tools/musa_unified/cmake/`包含此前额外CMake模块，避免继续依赖个人实验目录。Rust/vcpkg、源码和编译缓存是**构建依赖，不进运行包**。

镜像依赖工作树不一定等于远端gitlink版本。`seed_image_deps.py`是显式复用镜像版本，不宣称完成了干净递归clone的全依赖验收。不能用新的Mooncake源码整套替换而不验证接口。

## 3. xllm_pack到底需要什么

本工具生成的是基于固定镜像的**冻结运行包**，不是重新打包整个Torch/MUSA环境：

| 内容 | 是否必须有 | d1029-opt3已有？ | 包内位置 |
|---|---|---|---|
| 已验证P/D引擎 | 是 | 默认build/lib产物不是本次已验收版本 | `prefill/xllm`、`decode/xllm` |
| P/D各自Asio | 是 | 必须和引擎配对，不可任选镜像旧库 | 各角色目录`libasio.so` |
| DeepSeek专用AOT | 是 | Qwen目录缺少Model1/HC等专用目录 | `cached_ops/` |
| Qwen AOT | 是 | 有；冻结包仍带匹配副本 | `qwen_cached_ops/` |
| MATE MUBIN | 是 | 有0.2.7；冻结包仍带匹配副本 | `mate_mubin/` |
| DeepSeek修补版Control | PD需要 | 不等于镜像新版Qwen Service | `service/deepseek/xllm_master_serving` |
| DeepSeek Control的Rust tokenizer库 | 是 | 镜像Qwen那份hash不同，保留原配库 | `service/deepseek/librust_tokenizers.so` |
| Qwen Router/Coordinator Service及Rust库 | Qwen PD需要 | 有；冻结包带对应副本 | `service/qwen/` |
| etcd | 两套PD都需要 | 有；冻结包带副本 | `etcd/etcd` |
| MUSA hooks初始化shim | 是 | 有；冻结包带副本 | `bootstrap/libmusa_backend_init.so` |
| 启动、停止、测试脚本和profile | 是 | 原脚本可能依赖旧路径/config | `tools/`、`scripts/logger.py` |
| Qwen draft vocab | MTP需要 | 有；随工具目录带入 | `tools/musa_qwen3_8_vl_bench/draft_vocab_65536.txt` |
| 模型、tokenizer文件、图片和测试数据 | 运行/测试需挂载 | 部分测试资源在镜像内，模型在共享盘 | **不打进pack** |

Torch/Torch-MUSA 2.9.1、MATE/MUBIN 0.2.7、TVM-FFI `0.1.11.post1+musa.1`、TileLang-MUSA `0.1.12+musa.2`、muDNN/MCCL/MUSA SDK及其系统库沿用镜像，**不要从旧镜像整目录覆盖**。运行方式为独立C++ engine；本次不要求新打一个xLLM Python wheel，Python模型执行器/pybind wheel不是本次验收范围。

Custom AllReduce已经由源码`custom_all_reduce_ops.cu`编入引擎并显式注册，不再额外要求旧r13b独立`.so`。Mooncake Transfer Engine也随引擎构建；Python Mooncake `engine.so/store.so`不是Asio的替代品。

如果只做最小增量包，镜像已有的Qwen AOT、MUBIN、Qwen Service、etcd和shim可不重复带，但必须校验固定镜像对应文件。**本工具默认选择完整冻结这些小于SDK的依赖，避免跨目录混用；实际体积看`--plan-only`报告，不猜压缩率。**

### 打包命令

```bash
REPO=/你的Git仓库
BUNDLE=/data/nfs_shared/jd/d1029_compare_20261031/unified/bundle
DS_SERVICE=/data/nfs_shared/jiaming.wang/pd_v128_30k_20261005/stability_fix_20261007/control_relocation_20261008/build_r1/xllm_master_serving
# 这份是从已验证旧runtime取出的Control配套库，不是Qwen那份：
DS_RUST=/data/nfs_shared/jd/unified_toolkit_20261031/legacy_control/librust_tokenizers.so

cd "$REPO"
# 先只看清单、来源、hash和未压缩字节数，不生成安装目录。
bash tools/musa_unified/pack.sh --bundle "$BUNDLE" \
  --deepseek-service "$DS_SERVICE" --deepseek-tokenizers "$DS_RUST" \
  --output /data/releases/xllm_pack_d1029 --plan-only

# 新目录、新tar；不会覆盖已经安装的pack。
bash tools/musa_unified/pack.sh --bundle "$BUNDLE" \
  --deepseek-service "$DS_SERVICE" --deepseek-tokenizers "$DS_RUST" \
  --output /data/releases/xllm_pack_d1029 --tar
```

tar顶层为`xllm_pack/`，包含manifest及SHA256。`check_pack.py`校验Python包版本、全部文件哈希、两个引擎和两个Service的`ldd -r`与完整help XML；这些是依赖检查，**不是模型验收**。无需向`/usr/lib`全局安装旧Torch/FFI库。运行前先解到新目录并校验，部署系统按角色选择路径；此布局不是未经适配就能直接替换任意旧9n `bin/xllm`布局的承诺。

## 4. Qwen：保持熟悉的启动/测试方式

在健康的单机8卡容器内（此前使用94），使用仓库工具或pack中的同名工具：

```bash
export XLLM_PACK_ROOT=/opt/xllm_pack
export VL_PROFILE=d1029_unified
export VL_HOST=本机可达IPv4地址
export VL_PROMPTS=/root/xllm_vl_bench/vl_prompts_text20k_pm1k.json
export VL_WORK_DIR=/data/xllm_runs/qwen_d1029

python3 "$XLLM_PACK_ROOT/tools/musa_unified/check_pack.py" --pack "$XLLM_PACK_ROOT" --verify-hashes
cd "$XLLM_PACK_ROOT/tools/musa_qwen3_8_vl_bench"
bash start_stack.sh

bash check.sh --count 68 --max-tokens 256 --output "$VL_WORK_DIR/qa68"
mkdir -p "$VL_WORK_DIR/results"
bash bench.sh 6 --isl 22035 --shared 16384 --timeout 180 \
  --out "$VL_WORK_DIR/results/c6_first.json"
bash bench.sh 6 --isl 22035 --shared 16384 --timeout 180 \
  --out "$VL_WORK_DIR/results/c6_warm.json"
python3 "$XLLM_PACK_ROOT/tools/musa_unified/audit_qwen.py" \
  --result "$VL_WORK_DIR/results/c6_first.json" --result "$VL_WORK_DIR/results/c6_warm.json" \
  --prompts "$VL_PROMPTS" --server-log-dir "$VL_WORK_DIR/runs/current" \
  --qa-dir "$VL_WORK_DIR/qa68" --out "$VL_WORK_DIR/results/audit.json"
bash stop_stack.sh
```

`VL_PROMPTS`必须`export`或放在`bash bench.sh`同一行前缀；普通shell赋值不会自动传给子脚本。不默认指定`mlx5_2`，网卡按本机实际环境选择。

此profile固定K2、Graph、原镜像已验证配置，跳过个人`/workspace/config.json`。单机3P(CP2)+2D(TP1)：P用`0,1 / 2,3 / 4,5`，D用`6 / 7`；不是单实例TP8。实际缓存16384/约22000约74.47%，不是DeepSeek的80%。QA的256输出与性能的100输出分开；检查完整正文、usage/cache、服务错误和新Graph capture，不只看HTTP 200。

新profile的身份记录会等待exec后的非空cmdline连续稳定，再保存boot ID、PID启动时间、exe和argv hash。停止只操作本轮记录的进程；同一存活进程的身份异常变化会报错，不会静默跳过后宣称停止成功。不会按裸PID、进程名批量kill或reset GPU。默认旧profile保持原行为。

## 5. DeepSeek：2P1D+PCP启动与性能测试

准备三台健康8卡机器；Control可与P或D共用宿主机。**当前90 GPU3不适合做验收，不要直接照搬历史90D地址。**

将同一个pack部署到各容器，复制`tools/musa_unified/deepseek/cluster.example.json`为独立`cluster.json`，填写三个P0/P1/D地址、Control地址、模型、pack路径与日志路径。两个P各CP2×TP4，D为TP8；DSpark K6、Graph、PULL、prefix cache参数集中在`profile.json`，来源于旧版已验证配置。新包的新PD验收仍待完成。

```bash
KIT=/opt/xllm_pack/tools/musa_unified/deepseek
CFG=/data/config/deepseek_cluster.json

# 先在各自机器检查生成的本地命令，不启动进程。
bash "$KIT/start_stack.sh" control --config "$CFG" --plan
bash "$KIT/start_stack.sh" decode --config "$CFG" --plan
bash "$KIT/start_stack.sh" prefill0 --config "$CFG" --plan
bash "$KIT/start_stack.sh" prefill1 --config "$CFG" --plan

# 分别在对应的机器/容器执行，顺序不能颠倒。
bash "$KIT/start_stack.sh" control --config "$CFG"
bash "$KIT/start_stack.sh" decode --config "$CFG"
bash "$KIT/start_stack.sh" prefill0 --config "$CFG"
bash "$KIT/start_stack.sh" prefill1 --config "$CFG"
```

每次start只操作本机该角色，要求GPU/端口空闲，不自动停旧任务；生成独立run目录、实际argv/env、二进制hash、PID身份和日志。Control阶段检查etcd健康及RPC监听，不等待尚未启动的P/D所提供的模型HTTP接口；P/D阶段再检查各自HTTP。角色就绪不等于全链路正确，下一步必须跑严格测试。启动中途失败也保留`current`指向本轮，便于按身份记录清理。

```bash
export DS_MODEL_PATH=/data/nfs_shared/models/DeepSeek-V4-Flash-0731-FP8-mt
export DS_MODEL_ID=DeepSeek-V4-Flash-0731-FP8-mt
KIT=/opt/xllm_pack/tools/musa_unified/deepseek

# CPU先构建冻结请求，不调用模型；输出目录必须原本不存在。
bash "$KIT/bench.sh" 1,16 --isl 20000 --osl 100 --shared 16000 \
  --url http://CONTROL_IP:24888 --requests 32 --rounds 2 \
  --out /data/results/dsv4_20k100_b1_b16 --prepare-only

# 同一份请求，不允许执行后再悄悄重试或改输入。
python3 "$KIT/bench.py" --execute-prepared --out /data/results/dsv4_20k100_b1_b16

# 也可一步准备并执行，使用另一个新目录：
bash "$KIT/bench.sh" 1,2,4,8,16 --isl 20000 --osl 100 --shared 16000 \
  --url http://CONTROL_IP:24888 --requests 32 --rounds 2 \
  --out /data/results/dsv4_20k100_sweep
```

脚本验证真实chat模板下**20000输入**、**16000 cached_tokens**、**100输出**、EOS/finish/SSE完整性及正文严格`1..50`。每个并发先预热，再测；完整请求、响应、时间戳、tokenizer/输入hash留档。失败即停止后续阶段，不过滤错题后宣传性能。当前输出契约仅支持100输出；不要把它当作30K/1024或GSM8K验收。新Graph capture需结合服务日志判断，脚本不虚构capture=0。

需要停止时，分别在各机器执行（先P/D，再Control；Control会最后停止自建etcd）：

```bash
python3 "$KIT/run.py" stop --role prefill0 --config "$CFG"
python3 "$KIT/run.py" stop --role prefill1 --config "$CFG"
python3 "$KIT/run.py" stop --role decode --config "$CFG"
python3 "$KIT/run.py" stop --role control --config "$CFG"
```

## 验证边界

本工具不更改模型权重、SDK、驱动或主机GPU状态，不携带密码。源码、脚本、配置模板和文档进Git；ELF、AOT、MUBIN、模型、原始环境/响应日志和编译缓存不进Git。端到端验收进度以对应实验记录为准，不能把prepare-only、help或Control启动通过写成PD精度/性能通过。
