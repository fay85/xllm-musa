# 新 MUSA 镜像依赖、适配与验收记录（2026-10-29）

## 验收边界

当前任务：在 90/94/96 的 `xllm_musa_mt_main` 中恢复 DeepSeek-V4 DSpark 2P1D+PCP，比较严格 20K/100、80%前缀命中性能，并跑统一 GSM8K。

当前已修复 Gate 接口、补齐HC AOT，严格20K/100的TPOT已恢复至9.61ms。TTFT仍约550ms；r14补齐HC后的GSM8K前20题也是20/20，完整1319题已启动，不能宣布全量验收完成。

旧基线：B1 TTFT 444.150 ms、TPOT 9.839 ms，6/6严格通过；GSM8K 1268/1319 = 96.1334%。这是旧环境成绩，不是本轮结果。

| 新镜像配置 | TTFT平均 ms | TPOT平均 ms | 正式严格输出/cache |
|---|---:|---:|---|
| r12，MATE接口修复，custom AR未启用 | 661.973 | 18.008 | 6/6；全部16000缓存 |
| r13b，仅Decode恢复custom AR | 614.279 | 17.345 | 6/6；全部16000缓存 |
| r14，补齐HC AOT | 549.974 | 9.613 | 6/6；全部16000缓存 |

r13b 相比r12的TPOT改善约3.68%，不足以解释与旧9.839ms的全部差距。TTFT变化不能全部归因于Decode通信修改，需考虑窗口波动。

r14相对r13b，TPOT改善约44.6%，减少约7.73ms/token；r14与旧9.839ms基线已接近（本窗口略快约2.3%）。两轮P/D二进制和输入一致，变化仅为HC AOT的补齐，因此可以确认本次主要TPOT回退来自HC慢回退路径。TTFT仍比旧444.150ms慢约23.8%，需要继续定位。

r12 GSM8K诊断子集：原SGLang符号评分器 **20/20**，20条均正常stop、无请求错误/截断。补充numeric规则为19/20，不能替代主评分器；更不能把子集100%当作全量准确率。

r14同一诊断子集也为 **20/20**，全部正常stop、无错误/截断；对应生成目录 `gsm_smoke20_r14`，评分目录 `/data/nfs_shared/jd/xllm/logs/mt_main_smoke20_r14_20261029`。

```text
性能：/data/nfs_shared/jd/mt_main_eval_20261029_01/bench_b1_r12
      /data/nfs_shared/jd/mt_main_eval_20261029_01/bench_b1_r13b
      /data/nfs_shared/jd/mt_main_eval_20261029_01/bench_b1_r14
GSM20：/data/nfs_shared/jd/mt_main_eval_20261029_01/gsm_smoke20_r12
判分：/data/nfs_shared/jd/xllm/logs/mt_main_smoke20_r12_20261029
```

## 干净镜像检查方法

不仅检查已经修改过的开发容器，还临时启动了**没有 GPU、没有网络、没有共享盘挂载**的原始镜像容器，运行只读依赖检查后自动删除。

```text
镜像：registry.mthreads.com/presale/devtech/xllm:mt-d20261027-dev
ID：sha256:ad3833ac5f2a8e6d218b37a2b980fb54addff2de08ba310dda3a90de615e93cd
```

证据目录：

```text
/data/nfs_shared/jd/mt_feihu_build_20261029/mate027_r11/
  audit_mt_image_deps.py
  pristine_dependencies.json
  current_dependencies.json
```

## 编译与运行依赖清单

| 依赖 | 原始 Docker 是否有 | 当前使用位置/说明 |
|---|---|---|
| Python 3.10 开发头文件 | 有 | `/usr/include/python3.10/Python.h` |
| MUSA compiler / musamapping | 有 | `/usr/local/musa/bin/mcc`、`/usr/local/musa/tools/musamapping/mcc_wrapper` |
| CMake / Ninja | 有 | Python包版本 4.3.2 / 1.13.0 |
| Torch / Torch-MUSA | 有 | 均 2.9.1，`/usr/local/lib/python3.10/dist-packages` |
| MATE / MATE MUBIN | 有 | 均0.2.7；MUBIN在 `.../mate_mubin/mubin` |
| TVM-FFI | 有 | `apache-tvm-ffi=0.1.11.post1+musa.1` |
| TileLang | 有 | `tilelang-musa=0.1.12+musa.2`，使用split compiler/runtime |
| vcpkg及已安装依赖 | 有 | `/workspace/vcpkg-xllm`、源码build下 `vcpkg_installed` |
| Mooncake源码 / `libasio.so` | 有 | 镜像源码 `third_party/Mooncake`，build下 `mooncake-common/libasio.so`；本次P/D分别重编并配对保存 |
| yalantinglibs | 有 | `/usr/local/yalantinglibs/include` |
| OpenMPI | 有 | `/usr/local/openmpi` |
| MKL | 有 | `/opt/intel/oneapi/mkl` |
| MCCL / muDNN | 有 | `/workspace/MCCL_2.3.0/mccl/lib`、`/workspace/mudnn_3.4.0/lib`及MUSA库目录 |
| MUSA backend init bridge | 有 | `/workspace/artifacts/libmusa_backend_init.so` |
| 通用 GEMM AOT | 有 | `/opt/mate027_cache/mp31/cached_ops_jd/gemm_ops/gemm_ops.so`及grouped GEMM模块 |
| Rust / Cargo工具链及crate缓存 | **没有检查所需的路径** | 本次复用共享盘工具链及缓存，见下方 |
| DeepSeek `mla_ops` 与MODEL1专用AOT | **没有现成配套模块** | 新MATE wheel含源码；本次已构建新ABI模块，见下方 |
| DeepSeek HC专用AOT | 现成缓存中未找到 | r14已从新MATE源码构建，固定输入/Graph 5项检查、20K正式6/6通过；GSM8K待验收 |
| custom AllReduce 注册动态库 | 原始引擎未链接对应算子 | 本次从固定提交源码按新Torch/MUSA重编，并对齐Tensor引用签名；r13b Decode 8 rank确认启用 |
| MTTOplib | **没有，也不需要** | 当前分支使用MUSA/FlashInfer/MATE兼容路径 |
| 旧 `tilelang/lib/libtvm.so` | 原始镜像**没有** | 先前迁移额外拷入的旧库，不是缺失依赖；新loader不再加载它 |
| Control / etcd程序 | 镜像有 | `/opt/xllm_service/bin`；本轮Control实际沿用共享盘已验证的独立修补版，不能混淆 |
| 模型、冻结输入、GSM8K脚本/题库 | 使用共享盘资产 | 不属于本次引擎编译产物；评测入口见验收文档 |

共享盘复用项：

```text
Rust: /data/nfs_shared/jd/dspark_merge_20261016_1050/tooling/rust/stable-toolchain
Cargo cache: /data/nfs_shared/jd/dspark_merge_20261016_1050/deps/cargo-home
```

`/root/.cargo/bin/{cargo,rustc}` 在当前开发容器中的链接是此前补上的，不能据此声称原始镜像已带Rust。

`/opt/mate_0.2.7` 只有版本文件；wheel的实际编译资源在：

```text
/usr/local/lib/python3.10/dist-packages/mate/data/{include,csrc,mutlass}
```

## 本轮已解决的问题

1. 编译参数缺空格：SDK模块把Python include与 `-Od3` 连在一起。用独立CMake模块副本补空格，不改系统SDK。
2. 编译wrapper吞掉失败退出码：增加严格compiler/linker launcher并扫描日志，排除旧object造成的假成功。
3. Decode独立构建缺Python可执行文件与OpenMP路径：显式补齐，P/D编译与 `ldd -r`/`--help` 检查通过。
4. 旧、新TVM同时加载：P/D专用loader改用通用TileLang loader。旧顺序CPU对照退出134，新顺序通过且maps没有旧 `libtvm.so`。
5. 当前feihu源码缺DeepSeek PCP：仅导入可追溯的 `79980282` 与 `43ca4ce9`，保留新loader；未整树复制旧dirty worktree。
6. 新MODEL1 ABI：Decode增加9个scalar；KV view区分逻辑payload与物理page stride；Prefill改接combined-KV新ABI。新模块URI带 `mate027`，避免误加载旧参数签名。

算子对照：

| 检查 | 结果 |
|---|---|
| Prefill：1/7/59/60行，KV128/129/256/2049 | 4/4通过，最大绝对误差约0.0042 |
| Decode：1/7行、跨128页边界、padding stride | 2/2通过，最大绝对误差约0.0055 |
| Decode Graph：同图改变有效长度4→7→8 | 6/6通过，最大绝对误差约0.0063 |

这些是固定输入算子测试，不能代替整模型或GSM8K验收。

## 整模型首次尝试与 Gate 修复

运行目录：`/data/nfs_shared/jd/mt_main_eval_20261029_01/runtime_r11`。

Control、Decode、两个Prefill均注册就绪，但第一个20K primer在P94失败：

```text
Mate BF16->FP32 GEMM fallback: gemm_ops does not export bmm_fp16
Check failed: logits.defined() Required Prefill Gate projection unavailable
```

只尝试1个请求，0个通过；没有以失败请求的时间作为性能结果，没有重跑覆盖原日志。
本项目P/D按PID、start ticks、exe清单停止，未停止其他服务，未reset GPU。

新版 `gemm_ops` 已有 `bmm`，但接口与旧 `bmm_fp16` 不同。修复使用新符号，传连续权重并显式 `trans_b=true`，输出保持FP32，不降为BF16。

Gate独立检查：7/4096/8000行均通过；最大误差约1.0e-5，TopK发生变化的行数为0。

候选与证据：

```text
/data/nfs_shared/jd/mt_feihu_build_20261029/gate027_r12/
  gate027.patch
  test_mate027_gate.py
  gate_test.log
  bin/{prefill,decode}/
```

下一步：新候选完整启动 → 固定20K primer/warm/正式窗口 → 确认所有rank仅加载新运行库 → GSM8K全量。尚未把本轮标记为性能或精度验收通过。

## TPOT回退的后续定位

1. custom AR 起初缺算子注册；补库后又暴露C++ `Tensor` 与 `Tensor&` 签名差异。已修复独立插件，并通过C++ typed dispatcher检查。r13b 的8个Decode rank均有实际enabled日志，不是只看启动参数。
2. 新trace发现 `dsv4_hc_pre_decode_kernel` 累计55.806ms，约占采集到的155.514ms GPU kernel累计时间的35.9%。源码确认HC AOT缺失会直接进入这个本地回退kernel。Graph开着不等于选择了快算子。
3. 这些是可见kernel的累计时间，不能直接当作TPOT严格分解；Graph内部覆盖和多stream重叠均有限制。随后r14端到端A/B已验证TPOT由17.345降为9.613ms。

```text
trace：/data/nfs_shared/jd/mt_main_eval_20261029_01/runtime_r13b/d90/xllm_rank0_156303_1793254637993.pt.trace.json
采集控制：/data/nfs_shared/jd/mt_main_eval_20261029_01/profile_r13b_d90
HC新产物：/data/nfs_shared/jd/mt_feihu_build_20261029/hc027_r14
```

采集通过worker RPC进行，StartProfile/StopProfile均返回ok；profile请求严格输出通过，但该请求耗时不计入性能表。

r14不改变P/D二进制和模型参数，只增加同一新镜像MATE 0.2.7编译的HC prenorm及finish AOT。测试覆盖1/7/500行及Graph replay；没有重新开启额外的实验性MHC+Norm开关。

r14 runtime：`/data/nfs_shared/jd/mt_main_eval_20261029_01/runtime_r14`。两个P与D日志均确认Mate HC快路径实际启用。primer 2/2、warm 2/2、正式6/6通过；正式TTFT范围542.89–558.55ms、TPOT范围9.583–9.676ms。所有正式请求prompt=20000、completion=100、cached=16000，正文严格匹配。

环境注意：96宿主机系统盘已满，曾导致启动命令的宿主机 `/tmp` 输出写入失败。8 rank实际已启动，进程清单和日志在共享盘；后续输出已改写共享盘，未删除其他用户数据、未reset GPU。

## r14 全量GSM8K运行

引擎保持 `runtime_r14` 不变，客户端和原官方CPU评分器在 `90/wjm_ds910` 中顺序执行；客户端不使用GPU。生成仍使用原冻结runner，无重试、无删题、无参考答案输入。整轮生成有90分钟上限，超时会保留失败状态，不会被评分为完整结果。

```text
开始标记：/data/nfs_shared/jd/mt_main_eval_20261029_01/full_r14.started
日志：    /data/nfs_shared/jd/mt_main_eval_20261029_01/full_r14.log
退出码：  /data/nfs_shared/jd/mt_main_eval_20261029_01/full_r14.exit_code
生成：    /data/nfs_shared/jd/mt_main_eval_20261029_01/gsm8k/full_standard
最终评分：/data/nfs_shared/jd/xllm/logs/mt_main_full_r14_20261029
```

只有退出码0、完整1319条证据和官方评分汇总均通过核验后，才能给出全量准确率。当前尚无全量结论。
