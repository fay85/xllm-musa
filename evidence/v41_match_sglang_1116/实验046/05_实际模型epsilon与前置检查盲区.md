# E46 实际模型参数与前置检查盲区

ON v2没有产生正式性能数据，不能评价快慢。

实际权重视图config.json的text_config为rms_norm_eps=1e-20、hc_eps=1e-6、hc_mult=4、hc_sinkhorn_iters=20；配置SHA256为8072a73cadd4df137cb83777de9ba5ece3062912fb8b299a04a28533e0df7f9c。先前导出工具、测试fixture构造和dispatcher检查把RMS epsilon也写为1e-6，因此虽通过64+24+30项，却没有覆盖生产模型的完整数值参数。真实模型构造中的严格检查捕获了这一不匹配。不是MATE不支持，也不是需要更改模型权重；不删除保护检查，更不修改模型epsilon让它迁就候选核。

D全8rank自行exit-6，P主rank随后因为D侧etcd已经退出而连接失败，其余7rank正常exit0；两台10:44:55确认显存释放，未SIGKILL或设备复位。失败原始包18afd652已保留。

修正：导出工具和算子验证器直接读取模型config并记录SHA；RMS用1e-20，HC用1e-6，新增1e-10幅度输入检查；原生模型构造时核对AOT manifest的EPS、HC_EPS、Sinkhorn次数和split数。按实际参数重新导出AOT、编译原生模块、重新做数值/Graph检查，再以同一full_v3 ELF重新建立OFF/ON正式对照。旧OFF v2可作为历史环境记录，但不拿它与新ELF拼接成同二进制试验。

本轮教训：真实权重fixture不足以代表真实配置；参数化算子验证必须同时固定权重、配置、布局和模式。旧1e-6通过记录仍原样保留，但不能证明1e-20生产组合合格。
