# Graph兼容检查

目标：使原生模型在冻结SGLang运行环境中使用完整Graph，作为后续真正DP8/DeepEP LL接入的前置。不是恢复Python模型。

- 实际TorchMUSA2.9.0和2.9.1头均通过编译期能力断言及析构实例化。
- 2.9.0缺少allocator引用计数查询，故显式禁止自有VMM pool，普通native Graph pool由库本身管理；2.9.1原引用计数核对保留。
- 单卡运行：85次动态输入输出精确比较、23次registry与释放检查通过；两个图共存、refresh后继续回放、20轮反复创建释放及重复reset均覆盖。
- 模块SHA：a2df6e6ee06e8196ed5bc434bf4a2f8bee448342d42661c7ce4a3c9b062ddbc4。
- 原始文件：NFS builds/native_sg438_control/graph_check/runtime_v1/；GPU测试容器已正常TERM/exit0归档删除。
- 未完成：完整原生服务在新环境的基本题、实际token长度校验、正式性能和独立trace。当前最佳仍E34。
