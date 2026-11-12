# DEVICE首次启动失败：Mooncake端口绑定

## 事实

2026-11-12 16:13:45.531，SJ9 rank7在模型权重全部加载后、Mooncake初始化时出现`Fail to listen 0.0.0.0:42307`，继而`Failed to initialize MooncakeTransferEngineCore`并abort。rank0随后得到该worker的AllocateKVCacheWithTransfer RPC EOF，初始化失败。

客户端就绪轮询检测到服务已terminal，明确退出；没有发送基础题、没有正式测速，也没有进入V41/step。因此这次失败不能判定为设备event API或模型数值错误。

其余进程正常TERM退出，SJ9最终[-6,0,0,0,0,0,0,-6]，SJ10全8为0。16:14:50两端8卡都0MiB、无GPU进程，没有SIGKILL、重启或驱动改动。

## 已知原因与未证实部分

直接失败点是42307绑定失败。该地址属于当前P的transfer端口42300~42307，落在宿主默认临时端口区间内；当时具体占用者未在绑定失败瞬间抓到，不能宣称已证明是哪一个旧进程或TIME_WAIT导致。后续核对ss与实际bind探针，保存检查回执；不杀占用者、不修改全机临时端口策略。

## 处理

将P端我方服务固定端口从42000/42100/42200/42300体系改到经核验空闲、低于临时端口起点的22000/22100/22200/22300体系，D保持32000等原体系。同二进制、权重、参数与请求，独立ToDoList_3重跑DEVICE；引用本目录已有OFF/HOST作为观测开销参考。端口配置差异公开记录，跨轮TTFT波动不简单归因event。

故障tar.gz 201983字节，SHA573d4674973632382b041a3e7c80dff31cda9a71fd9143150304b81b54a825d7，已收回核验，位于03_正确性/DEVICE_failure。
