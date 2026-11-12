# E52 v1启动控制失败：未启动GPU rank

先前E52预检碰到正常退出后的GPU上下文短暂释放延迟，已只读等待归零。随后v1首次准备的sj10.json在SJ9侧已写入，但SJ10容器初次读取返回FileNotFound；D监督器在读配置阶段退出，P等待控制平面120秒后退出。两端都未创建rank0.log/status，P result.exit_codes为空；没有任何GPU rank启动，不是模型或算子报错。

后续只读检查两端容器的prepared配置均已可见，宿主/容器bind路径相同；这与NFS新文件/目录可见性时序一致，但不据此宣称已定位NFS内部根因。未改挂载、内核、驱动或Docker服务。只增加启动前两端容器内配置SHA核验，读到同一份完整配置才允许启动rank。

v1日志和空rank状态保留；另用t12_queued_c2_final_v2重试实际模型测试，避免覆盖失败目录。没有SIGKILL，没有重跑已经完成的性能数据，因为v1没有执行模型。
