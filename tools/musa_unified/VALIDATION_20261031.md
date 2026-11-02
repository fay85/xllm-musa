# d1029统一工具与pack验证记录

本记录区分编译/依赖验证、Qwen实际推理和尚未完成的DeepSeek PD验收。

## 构建与静态检查

- 新`build.sh`复用现有P/D缓存成功，两角色均`no work to do`；ELF SHA256未变化。
- 使用全新build目录、仓库内CMake模块，P/D两个角色均完成configure-only。没有把它标成全新空依赖的完整重编。
- Ruff 0.14.0检查和格式化通过；C++使用clang-format 20.1.6及本仓库配置，新增空白检查通过。
- 工具CPU回归15/15：拓扑/rank端口、Control RPC就绪、PID重用/exec空cmdline、实际记录与停止自建进程、清单自引用、预热顺序、错误缓存/正文拒绝等。
- 五图输入构造7/7单测和68条冻结提示结构校验通过。
- DeepSeek prepare-only生成70条20K/100请求（B1/B16示例、32条正式请求），实际输入严格20000；共享前缀LCP16000～16001，对齐后期望缓存严格16000；输出契约严格`1..50`。没有发送推理请求。

## pack迁移验证

在94新建相同image ID的容器，只挂载：

1. 模型目录，只读；
2. pack，只读，容器内`/opt/xllm_pack`；
3. 本次日志目录，可写。

**没有挂整个`/data`或历史源码、build、runtime目录。**两引擎、两Service均通过文件哈希、包版本、`ldd -r`和完整help XML检查。Asio、AOT、MUBIN、hooks shim及两个不同的Rust tokenizer库分别配套。

固定产物：

```text
P 097585d0cbcdb9cf6349067a595e4d1faa2d33e3ef5e07dbb6d9f40c3de0203e
D 31677d967f016abc928d7df42ee7d29a1c1b04d2c8590766c60395b7471a80d9
```

### Qwen

94单机8卡，3P(CP2)+2D(TP1)，K2、Graph开启、自动选择网卡。性能协议为C6、20K±1K文本加5图、100输出、shared16384。

| 验证 | 结果 |
|---|---|
| 启动 | 新脚本约72.6秒返回READY |
| 五图QA，256输出上限 | 68/68 |
| 两轮各60条性能请求 | 120/120完整正文、实际token数、缓存与finish门禁通过 |
| 第二轮TTFT均值 | 546.0 ms |
| 第二轮TPOT均值 | 15.17 ms |
| 第二轮新Graph capture / 服务错误 | 0 / 0 |
| 停止 | 身份校验停止成功，GPU进程全部退出、显存释放 |

此前相同联合引擎在原目录的测量为532.0/15.22 ms，本次约+2.63%/-0.33%，属于基本持平；不是性能优化收益，也不是多个模型/协议之间的横向比较。实际缓存比例约74.47%，不能写成80%。五图68/68仅指这组用例，不是通用图像准确率100%。

### DeepSeek

只完成了**Control及独立etcd的CPU启动验证**：配套Rust库加载、DeepSeek V4 chat template初始化、etcd健康返回200、Control RPC监听、按身份停止均通过。

Control必须先于引擎启动，因此Control阶段不等待模型HTTP接口；工具将此状态明确写为`readiness_kind=rpc`、`inference_verified=false`。**新包没有完成DeepSeek完整2P1D+PCP、性能或GSM8K验收。**

90 GPU3在主机重启后仍能独立复现显存读写错误，未用于本轮GPU模型测试。没有reset GPU、没有停止92的其他业务。

## 测试过程中修正的工具问题

- exec瞬间`/proc/PID/cmdline`可能为空。初版身份记录过早，导致安全检查跳过三个尚存进程；保留原记录后，用已核对的boot/start/exe、停止前完整argv及本轮日志FD精确回收。最终版要求非空cmdline连续稳定，并在同一存活进程身份变化时明确报错，新增测试后重新完整验证。
- 统一示例安装路径为`/opt/xllm_pack`；Control就绪改为独立的RPC门禁，避免启动顺序依赖尚未启动的P/D。
- 性能审计不只看HTTP200：逐请求匹配P/D日志，核对输入、16384缓存、100输出、finish以及异常正文；存在新capture的窗口单独标注。

## 证据与包大小

共享证据根目录：`/data/nfs_shared/jd/unified_toolkit_20261031/`。

- `build_check.log`、`build_check_engine_manifest.json`：现有缓存复用。
- `configure_probe.log`、`configure_logs/`：新目录配置检查。
- `prepare_20k_100/`：冻结请求和精确token构建。
- `smoke94_final/`：隔离容器mounts、完整QA/性能回复、实际进程/库映射、Control及停止记录。
- `smoke94_final/body_token_cache_audit.json`：120条性能请求和68条QA的最终审计。

冻结包约1026.6 MiB未压缩，gzip约354 MiB；如果严格依赖原镜像已有文件，只计镜像以外的组件约811.5 MiB。确切文件大小和SHA256以最终包manifest为准。模型、SDK、Torch/TVM-FFI/TileLang wheel、vcpkg、Rust和编译缓存没有打包。

后续在健康机器完成DeepSeek全链路回归前，不应将这个记录升级为DeepSeek性能/精度验收报告。
