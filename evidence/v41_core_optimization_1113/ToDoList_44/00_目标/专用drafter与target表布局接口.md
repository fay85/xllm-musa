# T14：原生drafter显式选择SWA表

当前target的NativeCausalLM按[SWA,C1,C2]解释multi_block_tables，且普通1P1D已验证实际为三组。DFlash/DSpark从target输入复制proposal，kv_split1时不展开物理表；专用drafter只有SWA，原来的“必须只有1张表”检查过窄。本候选明确支持纯draft输入的[SWA]，以及从target复制的[SWA,C1,C2]两种已知布局；始终使用第一张SWA表，检查二维且单请求，其余两张表由target持有，不读取也不写入。不是接收任意未知布局或删除全部检查。

P/D开启DSpark必须同步配置num_spec5和native标志，并通过bootstrap及draft context KV的实际PD传输核验。此前组件检查不能替代这一步。本修改只是模型接口接线，尚未完成整模型DSpark功能或性能，不晋级。

仅通过基础功能检查，未完成模型精度验收。
