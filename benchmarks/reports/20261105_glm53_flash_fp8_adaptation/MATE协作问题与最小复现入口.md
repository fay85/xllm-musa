# MATE 协作问题与最小复现入口

以下是供后续沟通使用的整理稿，尚未替用户发消息或提MR。当前不因等待回复中断已能运行的正确性基线。

## 优先确认：GLM clamp 的 decode 路径

现场版本MATE `0.2.8.post1`，Torch/TorchMUSA `2.9.1`，MUSA SDK `5.1.0`。sj9/sj10的194个MATE源码/so组合SHA256相同：`94ec263d988f36308d24b929b06c9f53dede3b8ef1dd80f9139fb0979d9504f2`。具体文件hash在`参考材料/dependency_hashes_sj9.json`。

安装位置`/usr/local/lib/python3.10/dist-packages/mate/mega_moe.py`：`fp8_fp8_mega_moe`（738行）向两个分支传activation_clamp；`_run_prefill_mega_moe`（702行）继续传入`_prefill_stage1`；`_run_decode_mega_moe`（652行）接收该参数，但675行调用stage1时没有继续传入clamp。行号对应本轮安装文件，不能套用到其他版本。

GLM-5.3-Flash要求SwiGLU的gate上界10、up区间[-10,10]。目前我方显式把小输入补到129行，invalid route=-1、weight=0，使用支持clamp的split-prefill路径，只返回真实行；前后同步和Gloo barrier也保留。因此当前接入可运行，但单token开销较大。

可向同事提出：

> 我在xLLM的GLM-5.3-Flash接入了MATE原始FP8 MegaMoE，使用Attention TP8、路由专家EP8、共享专家TP8。当前0.2.8.post1的decode包装接受activation_clamp但没有继续传给stage1；为保留GLM的clamp=10语义，暂时补129行走split-prefill。请帮忙确认是否已有支持该clamp的decode实现或推荐版本，以及IPC工作区复用、stream依赖的同步契约。我们已有EP2/EP8、1/16/65/129 token、大幅值clamp及真实E288/H4096/I2048权重回归，可以用来验证新路径。

## 复现入口与范围

- 仓库：`fay85/xllm-musa`的`xiaopang1104benchmark1P1D`，本轮最终提交前以source_manifest_v14c记录源文件身份。
- `tests/musa/run_glm_mega_moe_regression.py`：随机EP2/EP8、真实FP8量化、独立FP32公式、clamp大幅值、shared合成和连续调用。
- `tests/musa/run_glm_real_moe_regression.py`：GLM真实checkpoint第3层，E288/H4096/I2048/top8，分别对routed/shared/combined输出。
- 我方适配：`xllm/python/kernels_musa/mega_moe.py`，不将CPU oracle用于在线推理。
- 本地及NFS原始日志、版本清单保留，不能以测试阈值通过代替整模型/长上下文对分。

## 其他候选需求，尚不能断言库缺失

- mHC post和sigmoid-gated norm能否复用现有融合算子，先核对方程、精度和返回形状。
- indexed-state短卷积及graph固定工作区/IPC通信支持，需覆盖状态槽、COW、padding。
- 原生FP8 NoPE sparse MLA的cache布局迁移，需连同写入量化、page布局、P/D传输和精度一起改；当前BF16 cache的GPU参考路径已有语义实现。
- KDA chunk/recurrent与整模型形状敏感性需要固定同一token前缀逐层排查。现有KDA独立测试通过不能直接消除整模型数值反例。

当前没有基于推测修改MATE底层代码或升级共享驱动。替换新库后必须生成新版本账本，先做算子/通信回归，再重点题及固定题单验证，合格后复测性能。
