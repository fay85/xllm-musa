# T03：mHC 分裂投影归约的累计版本重试

沿用实验8已经通过的40+6例基本检查和同一个CUDA/MUSA归约kernel，不重新开发。当前候选保留已接受的TP/EP custom allreduce、SWA读融合、压缩读复用、shared TP及model stream；同时保留optional FFN tail Graph的代码，是否开启由其独立正式结论决定。Gate缓存仍未晋级，默认关闭。

只把MATE prenorm投影后的split partial / square-sum归约和RMS缩放合成一个kernel。P端开关固定关闭、D端单独开关；两组正式测速和对应trace必须在同一累计方案、同一ELF下完成。不能用先前4/5失败的长输入结果来晋级，也不能从微基准下降直接推算全局改善。

实验8的P端停滞并不能证明D端mHC kernel错误；本轮在P端计算流诊断后再启动。当前只是准备候选、未开始GPU整模型测速。

仅通过基础功能检查，未完成模型精度验收。
