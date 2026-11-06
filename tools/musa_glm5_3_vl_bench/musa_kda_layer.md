# MUSA GLM KDA attention 整层

实现：`xllm/core/layers/musa/glm5_next_kda.{h,cpp}`，构建入口为
`xllm/core/layers/musa/CMakeLists.txt::musa_layers`。本次在工作区已有草稿上修复和补齐，
未注册完整 GLM 模型：decoder、mHC、FFN/MoE、视觉塔仍是独立接入任务。

## 调用与参考

vLLM checkout `ebac032b92cc55c2232deced9217cd5122648ce8`：
`vllm/models/glm5next/common/kda.py::Glm5NextLinearAttention`。
按 q/k/v/b/f_a/g_a 顺序融合输入投影，前四项按头分片，后两项复制；
f_b/g_b 按头分片，o_proj 行并行归约。checkpoint 传入时由调用方去掉
`model.layers.N.self_attn.` 前缀。KDA 权重按 vLLM 保持 BF16，FP32 conv/gate 参数按目标 dtype 加载；拒绝 FP8 权重，不能忽略 scale 后直接转换。

仓库内部参考：MLU `layers/mlu/glm5_next/glm5_next_kda.cpp` 的投影和权重加载，
及 `glm5_next_decoder_layer.cpp` 的 self_attn 调用边界；MUSA
`qwen3_gated_delta_net_base.cpp` 的卷积和 framework cache 契约。
MUSA 的 conv 为 `[slots,3*Hlocal*128,3]` BF16；SSM 为
`[slots,Hlocal,128,128]` FP32，V-first。不照搬 MLU 的 conv 排布。

执行链：融合 Linear → Q/K/V causal conv + SiLU → MATE flash_kda →
FP32 RMSNorm × sigmoid(output_gate) → BF16 → RowParallelLinear。
Q/K/V 为 `[1,T,Hlocal,128]`，允许行 stride；raw beta/gate 在 MATE 内激活，
scale=1/sqrt(128)，gate=lower_bound*sigmoid(exp(A_log)*(raw_gate+dt_bias))。
无 RoPE 或 softmax mask；因果性由卷积及逐序列递推保证。

Attention 选择停在 MATE：现有 `kernels/musa/kda.cpp::flash_kda` 已有匹配 API，
因此不降级到 muDNN/muBLAS。复用已安装 MATE 的 prepare/recurrence TVM-FFI 模块、
workspace 与 stream guard，不增加 ABI。MATE checkout
`9d3260d633597f08fc9d5d4dc73344872241c34d` 的 `mate/kda.py::chunk_kda`
仅作为语义参考；运行导出模块来自本机安装版，不能把 checkout 当作安装版本。
Linear、conv、norm 使用现有 MUSA/TorchMUSA 封装，不引入 host 数据计算。

## 支持边界

BF16 activation、head_dim=128、Q/K/V 相同头数、conv width=4、CP=1、eager。
packed prefill、chunk continuation、普通 decode 共用 MATE core。
新请求必须提供 `has_initial_states=false`；续接为 true；decode 使用已初始化状态。
状态槽必须非负、唯一且在池内。明确拒绝 dummy、mixed、spec verify、graph capture、
checkpoint_stride!=1 和 distinct source slots；prefix 恢复必须先完成拷贝再调用。
验证 metadata 的标量同步仅用于 eager 输入校验，不是 host attention fallback。
普通 TP 的参数分片已实现，但多卡整层数值尚未验证。

## 验证（2026-11-06）

- `/usr/local/bin/clang-format` 20.1.6 格式化所有改动 C++；`git diff --check`。
- 新测试 `tests/core/kernels/musa/glm5_next_kda_layer_test.cpp` / target
  `glm5_next_kda_layer_test`：独立 FP32 逐 token 参考，packed 19/21 token，
  短 chunk、跨 MATE chunk 边界、decode、state 对比和槽复用 reset。
- 源码与测试通过独立 g++ C++20 编译（沿用保存的 CMake include/define 参数）。
- 已有算子二进制实机重跑 3/3 通过：

```bash
FLASHINFER_OPS_PATH=/opt/mate027_cache/mp31/cached_ops_jd_k8d \
MATE_DISABLE_JIT=1 build/lib.linux-x86_64-cpython-310/xllm/kda_test
```

上述是已有算子二进制，不代表新整层验收。

新目标标准构建尝试：

```bash
TORCH_MUSA_PYTHONPATH=/usr/local/lib/python3.10/dist-packages/torch_musa \
CPLUS_INCLUDE_PATH=/usr/include/python3.10 \
MUSAMAPPING_PATH=/usr/local/musa/tools/musaMapping \
cmake --build build/cmake.linux-x86_64-cpython-310 \
  --target glm5_next_kda_layer_test -j4
```

被构建目录重新配置阻塞：vcpkg baseline `fba75d09065fcc76a25dcf386b1d00d33f5175af`
无法读取 `versions/baseline.json`，`fatal: not a git repository: '.git'`。
随后生成目录缺少 `CMakeFiles/rules.ninja`。日志 `/tmp/glm_kda_build.log`。

独立对象与已有库链接时先遇到 `create_prefix_cache(PrefixCache::Options)` 未解析，
使用函数分段及 gc-sections 隔离未使用的模型执行路径后可链接；但程序进入 gtest
前即在 TorchMUSA 静态注册报 `ConvPackedParamsBase<2> ... could not be converted
 to any of the known types`（`/tmp/glm_kda_layer_test.log`）。因此整层数值测试
尚未执行成功，不能声称精度通过。独立编译脚本 `/tmp/check_glm_kda.py`、
`/tmp/build_glm_kda.py`，链接脚本 `/tmp/link_glm_kda.py` 保存本次复现参数。

需恢复一致的构建/依赖产物后重建并执行新 target。未完成真实权重整模型 HTTP 推理、
多卡 TP、长序列与吞吐验证；当前没有可供服务选择的完整 MUSA GLM native model。
