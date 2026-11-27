# mhcpresinkhorn_rmsnorm（v4-K + RmsNorm 融合）接口说明

## 1. 公开接口

可运行示例 `mhcpresinkhorn_rmsnorm_demo.py` 的完整源码见[设计说明](design.md)第 7 节，其中包含输入构造、模块加载与四个输出的 PyTorch 参考值对比。由于 `pypto-pro` 含连字符且不是 Python 包，示例按文件路径加载算子模块：

```python
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

path = Path("xllm/python/kernels_npu/pypto-pro/mhcpresinkhorn_rmsnorm.py")
spec = spec_from_file_location("mhcpresinkhorn_rmsnorm", path)
mod = module_from_spec(spec)
spec.loader.exec_module(mod)

y, rstd, post, comb = mod.mhcpresinkhorn_rmsnorm(
    x, hc_fn, hc_scale, hc_base, gamma,
    hc_mult=4, sinkhorn_iters=20, eps=1e-6,
)
```

### 1.1 输入

| 参数 | 形状及类型 | 含义及约束 |
| --- | --- | --- |
| `x` | `[t,4,4096]` bf16 | 超连接输入，对 hc 和 d 维计算逆 RMS；要求连续，`t >= 1` |
| `hc_fn` | `[24,16384]` bf16 | 门控矩阵权重，24 路为 4 pre + 4 post + 16 comb；要求连续。权重应在模型侧转换为 bf16，算子直接使用 BF16×BF16→FP32 矩阵乘，使用的权重与先将 fp32 权重转换为 bf16 再计算的路径相同，不代表与原始 fp32 权重数值位等价；bf16 权重存储将其 GM 流量减半 |
| `hc_scale` | `[3]` fp32 | 三条路径的 `[s_pre,s_post,s_comb]` 缩放值；要求连续 |
| `hc_base` | `[24]` fp32 | 三条路径各自的偏置，分别为 `[0:4]`、`[4:8]`、`[8:24]`；要求连续 |
| `gamma` | `[4096]` bf16 | RmsNorm 每维的缩放权重；要求连续 |
| `hc_mult` | 整数，默认 4 | 超连接倍数，当前仅支持 4，否则报错 |
| `sinkhorn_iters` | 整数，默认 20 | Sinkhorn 迭代次数，当前仅支持 20，否则报错 |
| `eps` | 浮点数，默认 `1e-6` | 门控及 Sinkhorn 的稳定常数，当前仅支持 `1e-6`，否则报错 |
| `rms_eps` | 浮点数，默认 `1e-6` | RmsNorm 的 epsilon。为签名兼容保留为入参，但只接受默认值；kernel 使用编译期常量 `RMS_EPS = 1e-6`（`rstd = 1/sqrt(mean(y²)+RMS_EPS)`） | 取其他值报错 |

包装函数检查输入形状、类型及 `x`、`hc_fn`、`hc_scale`、`hc_base`、`gamma` 的连续性，并要求 `t >= 1`；`x` 不在 NPU 上时直接报 `ValueError`。随后依据 `TILE_FWK_DEVICE_ID` 将输入移至相应 NPU，四个输出和内部工作区也都分配在该设备上；因此位于其他 NPU 的输入会被搬到目标设备，而不是被拒绝。实现未检查所有输入是否原本在同一设备。

### 1.2 输出

| 返回值 | 形状及类型 | 含义 |
| --- | --- | --- |
| `y` | `[t,4096]` bf16 | 已经完成 RmsNorm 的 `y_norm = y_bf16*rstd*gamma`；不返回未经归一化的中间 `y_bf16` |
| `rstd` | `[t,1]` fp32 | 根据 bf16 中间 `y` 计算的 RmsNorm 逆标准差，供反向使用 |
| `post` | `[t,4]` fp32 | 后置门控 `sigmoid(z)*2` |
| `comb` | `[t,4,4]` fp32 | Sinkhorn 归一化后的组合矩阵 |

## 2. 内部常量与切分

| 常量 | 数值 | 含义 |
| --- | --- | --- |
| `HC` / `D` / `HC_D` | 4 / 4096 / 16384 | 超连接倍数、隐藏维度及矩阵乘 K 维 |
| `MIX_HC` | 24 | 门控通道数：4 pre + 4 post + 16 comb |
| `SINKHORN_ITERS` / `EPS` / `RMS_EPS` | 20 / `1e-6` / `1e-6` | 默认迭代次数和数值常数 |
| `NB` / `TG` | 28 / 16 | MIX 逻辑块数 / 每组 token 数 |
| `MIXC` | 32 | 门控矩阵乘 tile 宽度（有效通道为 24） |
| `DC` / `NDC` | 512 / 8 | y 数据流分片宽度 / 分片次数 |
| `PWC` | 24 | pmix3 工作区行宽（有效门控通道数） |
| `SGRP` | 16 | 每个部分归约组的最大 token 数 |
| `NAIV` | 56 | AIV 数量；每个 token-flat owner 计算完整 per-token ssq |
| `NT` | 32 | Sinkhorn 批深度（一个 lane 一个 token） |
| `KA` / `KLAST` | 592 / 400 | 前 27 个 AIC 与最后一个 AIC 的 K 分片宽度 |

K 维覆盖为 `27*592+400=16384`，没有重叠或未覆盖片段。

## 3. 直接调用 Kernel（高级用法）

一般应使用公开包装函数；直接调用需要自行分配输出和工作区。包装函数的 `_get_workspaces` 按 (t, device, stream) 缓存：ACL graph 捕获期间创建的条目常驻，其余 eager 调用使用有界真 LRU（上限 8 条），多流不会共享同一份工作区。pmix3 额外有 64 行 token padding；r_ws 为 `[t,8]` fp32；comb16 按 `_comb16_rows(t)` 行分配（覆盖最后一个 chunk 的 AIV 窗口取整，例如 t=1024 为 1072 行，而不是 `NAIV*ceil(t/NAIV)` = 1064 行）。直接调用时工作区生命周期由调用方负责，当前 kernel 未完成前不得复用。

```python
mhcpresinkhorn_rmsnorm_kernel_k[None, 28](
    x_2d,      # [t,16384] bf16，x 的二维视图
    x_3d,      # [t,4,4096] bf16，原始 x
    hc_fn,     # [24,16384] bf16
    hc_scale,  # [1,3] fp32 视图
    hc_base,   # [1,24] fp32 视图
    gamma,     # [1,4096] bf16 视图
    y,         # [t,4096] bf16 输出
    rstd,      # [t,1] fp32 输出
    post,      # [t,4] fp32 输出
    comb16,    # [_comb16_rows(t),16] fp32，comb 的平铺视图（末个 AIV 窗口补齐）
    pmix3,     # [28,ceil(t/16)*16+64,24] fp32 工作区
    pmix3f,    # 二维视图 [28,(ceil(t/16)*16+64)*24]
    r_ws,      # [t,8] fp32，per-token 完整平方和
)
```

## 4. 运行环境

| 环境变量 | 用途 |
| --- | --- |
| `TILE_FWK_DEVICE_ID` | 目标 NPU 设备编号；默认 0 |
| `PTO_TILE_LIB_CODE_PATH` | 本地 `pto-isa` 检出目录，供 PyPTO-Pro 编译使用；本目录不包含该依赖 |

原开发环境使用 CANN 9.2.0，并通过 `/usr/local/Ascend/ascend-toolkit/set_env.sh` 加载环境；实际安装路径请以当前设备为准。示例调用会触发 JIT 编译，须安装兼容版本的 PyTorch、`torch_npu`、PyPTO-Pro 和 CANN。更多实现信息见[中文版设计文档](design.md)。
