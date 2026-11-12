# 三层 DSpark 计算主体组件回归通过

仅通过基础功能检查，未完成模型精度验收。

原生三层纯SWA drafter已编译，main_proj/main_norm一次投影并为三个draft层写context，anchor/noise query块采用非因果注意力；层间传递shifted pre_mix，最后ffn_pre collapse+norm，再词表head。它使用实际MATE稀疏注意力、mHC和W4A8专家；没有CPU前向回退。

12例覆盖默认/非默认流、上下文0/129/257、两组anchor及5宽queryblock。检查context投影与已验证组件相同、KV注入正确、proposal后历史缓存逐位不变、词表head与线性参考一致。采用实际layer0 mHC和两个专家权重fixture，其余投影/embedding/head为合成张量、单GPU切片几何；这不是TP8完整checkpoint验证。

实际完整checkpoint连接正在独立实验34完成：mt的main_proj保存为FP8+UE8M0，Markov键为embed/head而非旧通用类的markov_w1/w2。仍需target37/38/39 entering残差捕获、spec验证状态恢复和runtime工作流；当前尚无MTP性能收益，不晋级累计最优。

原始组件归档与SHA见03_正确性；source_delta_vs_best_4411.patch包含未晋级候选，不能误称其已在主分支启用。
