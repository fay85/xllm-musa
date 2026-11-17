from pathlib import Path
import datetime,json,sqlite3,re,statistics,shutil
W=Path.cwd();O=W/'王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/20261116后续如何正确推进并知道边界大概在哪里/追平sglang甚至超越它';C=O/'00_总控与恢复';E=O/'02_实验记录/实验_049';B=O/'03_累计最佳'
rows=[];validations=[]
for name,sub in [('ON1','full_draft_v1'),('OFF_reverse','reverse_control_v1'),('ON2','confirm_on_v1')]:
 p=E/('原始证据_'+sub);summary=json.loads((p/'formal_client/summary.json').read_text());db=next((p/'formal_client/formal').rglob('benchmark_data.db'));cx=sqlite3.connect(db);cx.row_factory=sqlite3.Row
 data=[dict(x) for x in cx.execute('select success,prompt_tokens,completion_tokens,latency,first_chunk_latency,time_per_output_token,request_id from result')];assert len(data)==5
 for x in data:
  assert x['success']==1 and x['prompt_tokens']==256 and x['completion_tokens']==256
  x['recomputed_tpot_ms']=(x['latency']-x['first_chunk_latency'])/255*1000
 assert abs(statistics.mean(x['recomputed_tpot_ms'] for x in data)-summary['mean_TPOT_ms_per_token'])<1e-8
 assert abs(1280/sum(x['latency'] for x in data)-summary['output_tokens_per_latency_sum_second'])<1e-8
 log=(p/'roles/sj10/rank0.log').read_text();srv=[]
 for line in log.splitlines():
  if 'sequence_terminal' in line and 'prompt_tokens: 256, generated_tokens: 256' in line:
   m=re.search(r'speculative_accepted_tokens: (\d+), speculative_proposed_tokens: (\d+)',line);assert m; srv.append([int(m[1]),int(m[2])])
 assert srv and all(x==[130,375] for x in srv)
 rows.append({'name':name,'summary':summary});validations.append({'name':name,'rows':data,'server_short_request_count':len(srv),'all_server_accepted_proposed':[130,375]})
(E/'03_性能').mkdir(exist_ok=True);(E/'03_性能/三组逐请求与服务端交叉核验.json').write_text(json.dumps({'groups':validations,'metrics':rows},ensure_ascii=False,indent=2)+'\n')
confirm=rows[2]['summary'];first=rows[0]['summary'];off=rows[1]['summary'];prev_fast=35.036918337
long=json.loads((E/'原始证据_confirm_on_v1/long_smoke/result.json').read_text());post=json.loads((E/'原始证据_confirm_on_v1/post_long_checks/results.json').read_text());assert all(x['passed'] for x in post)
assert long['usage']['prompt_tokens']==2048 and long['usage']['completion_tokens']==256
for h in ['sj9','sj10']:
 p=E/'原始证据_confirm_on_v1/roles'/h
 assert json.loads((p/'result.json').read_text())['exit_codes']==[0]*8
 assert json.loads((p/'resource_release.json').read_text())['gpu_free']
 shutil.copy2(p/'role_config.json',B/f'full_draft_sg438_{h}_role_config.json')
sha=confirm['binary_sha256'];now=datetime.datetime.now().astimezone().isoformat()
snapshot={'decision':'PHASE_PROMOTION_SMALL_GAIN_NOT_FINAL','run':'p49_full_draft_confirm_on_v1','source_commit':'9046ef0292aa7bff88d2c5bcf248714e278d9ec9','binary':'/expt/artifacts/native_sg438_runtime_timing_v1/bin/xllm','binary_sha256':sha,'source_manifest':'/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116/builds/e48_native_timing_v1/source_manifest.json','configuration':'E39 policy plus D FULL_DRAFT_GRAPH=1; AttentionTP8/EP8/sharedTP8, proposal3, targetq4 Graph, static expert activation; ownerLL/prequant/E46dot off','metrics':confirm,'first_on_metrics':first,'reverse_off_metrics':off,'long_smoke':{'usage':long['usage'],'E2E_s':long['latency_s'],'first_content_s':long['first_content_s'],'cold_capture_not_formal_performance':True,'post_long_checks_passed':True},'all16_normal_exit':True,'independent_trace16_valid':True,'final_three_rounds_done':False,'formal_long_1warm5_done':False,'same_window_sglang_done':False,'goal_achieved':False,'updated_at':now}
backup=C/'历史入口快照_E49晋级前';backup.mkdir(exist_ok=True)
for n in ['当前阶段最佳.json','当前累计最佳.md','当前最佳.md','README.md']:
 if (B/n).exists() and not (backup/n).exists():shutil.copy2(B/n,backup/n)
(B/'当前阶段最佳.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n');(B/'E49_完整draftGraph阶段最佳.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+'\n')
table='|配置|TTFT ms|TPOT ms/token|TPOT 标准差 ms|E2E s|输出 token/s|\n|---|---:|---:|---:|---:|---:|\n'
for item in rows:
 m=item['summary'];table+=f"|{item['name']}|{m['mean_TTFT_ms']:.6f}|{m['mean_TPOT_ms_per_token']:.6f}|{m['TPOT_sample_stdev_ms']:.6f}|{m['mean_E2E_s']:.6f}|{m['output_tokens_per_latency_sum_second']:.6f}|\n"
report=f'''# E49 完整 draft Graph：小幅收益，作为后续累计起点

仅通过基础功能检查，未完成模型精度验收。**本项阶段晋级，最终 goal 未达成。** 当前保守引用第二次 ON：TPOT {confirm['mean_TPOT_ms_per_token']:.5f} ms/token、输出 {confirm['output_tokens_per_latency_sum_second']:.5f} token/s、TTFT {confirm['mean_TTFT_ms']:.5f} ms。

## 为什么重试，以及实际改了什么

旧 E14 在旧环境中收益不明确。本次 E48 在当前 SG438 兼容环境、原生通信和 E39 静态激活下，测出 draft 边界约 5.68 ms/轮，提供了新的测试依据。本项仅把 D 的 `XLLM_DSV41_FULL_DRAFT_GRAPH` 从 0 改为 1，复用已实现的三层 draft body 捕获；没有另改模型方程、proposal 数或 target verify 行数。P 端保持原配置。

源码记录为 9046ef0292aa7bff88d2c5bcf248714e278d9ec9，三个配置用相同 ELF `{sha}`。构建清单为 NFS `builds/e48_native_timing_v1/source_manifest.json`，采样开关只用于独立诊断，正式测试没有 arm 文件。

## 正式数据和晋级理由

每组同 W4AFP8-mt1017、SJ9/SJ10 各8卡、Attention TP8/专家 EP8、实际256/256、C1；各自1暖机+5正式，profiler关闭。EvalScope1.14.1 原始 SQLite、HTML、工具吞吐和逐请求数据均保留。

{table}

前置 E48 的紧邻反向 OFF 是27.648372 ms/token、35.036918 token/s；更早 OFF 是28.183333 ms/token、34.010076 token/s。ON1 相对紧邻前OFF输出提高 {(first['output_tokens_per_latency_sum_second']/prev_fast-1)*100:.3f}%，相对后OFF提高 {(first['output_tokens_per_latency_sum_second']/off['output_tokens_per_latency_sum_second']-1)*100:.3f}%。ON2 相对后OFF提高 {(confirm['output_tokens_per_latency_sum_second']/off['output_tokens_per_latency_sum_second']-1)*100:.3f}%，相对前面最快OFF只有 {(confirm['output_tokens_per_latency_sum_second']/prev_fast-1)*100:.3f}%。

两次独立启动的 ON 都优于前后 OFF，变化方向一致；结合 draft 区间下降和实际图回放证据，将其作为小收益累计起点。**不宣称稳定提升4.31%，也不宣称统计显著或已经追平。** 每组只有5次且控制有漂移，后续最终三轮仍须重新完成；本次两次ON不能冒充三轮最终验收。

已从 SQLite 重算15条正式请求的 TPOT 和总token/总latency，全部 success=1、256/256；与rank0服务记录交叉检查，短请求均 accepted130/proposed375，未靠缩短输出或改变工作量取得成绩。详见[逐请求核验](03_性能/三组逐请求与服务端交叉核验.json)。

## 从图和计时看优化发生在哪里

独立事件采样的 rank0 draft 区间5.681639→4.530147 ms/轮，约下降20.27%；target+head47.000716→46.936303 ms/轮，基本不变。采样总边界54.703999→53.322481 ms/轮。这些边界含跨流等待，不是kernel busy时间，也不是HTTP总时延占比；不能用20.27%预测整请求提速。

![分段计时](04_trace/运行时分段对比.png)

16rank trace和原生sidecar核验通过。D每rank的target图仍4047节点，完整draft图277节点；相同采集窗口各有4次target和draft回放。三层draft body已入图，但词表头、Markov及其外围缓存操作并非因此全部入图。图节点不等于Python/CPU调用数，不能把少见CPU事件自动解释为少执行模型。

![target图](04_trace/E49_full_draft_graph/Perfetto图/真实CPU区间与GPU时间线.png)

![完整draft回放](04_trace/E49_draft_body_correlated/Perfetto图/真实CPU区间与GPU时间线.png)

最初按CPU提交区间截取draft图片只看到了很短的窗口，GPU异步节点被裁掉；原图保留并标注，正文采用GraphLaunch correlation关联的完整GPU窗口。没有延长CPU事件来伪造同步执行。正式性能与采集分开，重型trace不能直接用来排全局收益上限。

## 长输入、退出和证据

相同实现完成一次2048/256冒烟，usage核验通过，E2E {long['latency_s']:.6f}s、首有效内容{long['first_content_s']:.6f}s。该请求跨到新的图桶，含首次捕获，**不是暖机后的正式长输入性能**。随后两道题答出4.2/56，256/8回短也成功，未发现本次跨图桶缓存污染。正式2048/256的1暖5仍未做。

三次模型运行每次两台各8个rank均正常exit0；第二ON于12:24核验两台显存释放。无SIGKILL、GPU reset、驱动变更或他人任务操作。

NFS根：`/data/nfs_shared/xiaopang/v41_xllm_match_sglang_20261116`。三个run依次为 `p49_full_draft_graph_sg438_v1`、`p49_full_draft_reverse_control_v1`、`p49_full_draft_confirm_on_v1`；原始trace位于对应profiling目录，小日志和归档SHA在本地三个 `原始证据_*` 目录与 `归档回执_*.json`。ON2证据包SHA为a017caf298cec07c587fe6a318378dc9af14c1ed467c68ba178a8231f25c3158。

首ON formal identity误继承旧manifest路径，已以identity_correction.json保留原文件并明确更正；实际ELF核验正确。另发现累计最佳通用入口仍停留E34，而E39专页/状态已更新。本次把旧入口归档后统一至E49，防止后续从旧起点继续。所有历史结果保留。

## 下一步

先对来自V4的既有SIMT Gate GEMV做真实BF16权重、FP32累加/输出、动态Graph和小行微基准；它与E31负收益的MATE BF16→FP32 GEMM不是同一实现。只有实测支持才进入整模型，始终以E49为当前配置起点。owner/LL、FP8大投影等负收益/前置实现继续保留，不无新证据原样重做。

距离固定门槛，ON2的TPOT仍需降低 {(1-21.57/confirm['mean_TPOT_ms_per_token'])*100:.2f}%，输出仍需提高 {(39.45/confirm['output_tokens_per_latency_sum_second']-1)*100:.2f}%。最终同实现三轮、正式长输入、同窗口SGLang和收口条件都未完成。
'''
(E/'05_结论与下一步.md').write_text(report)
short=f'''# 当前累计最佳：E49 完整 draft Graph

当前继续开发的起点为 **E49**，继承E39静态激活并开启D完整draft body Graph。第二次独立ON为TPOT **{confirm['mean_TPOT_ms_per_token']:.5f}ms/token**、输出 **{confirm['output_tokens_per_latency_sum_second']:.5f}token/s**、TTFT **{confirm['mean_TTFT_ms']:.5f}ms**、E2E **{confirm['mean_E2E_s']:.6f}s**。第一次ON为26.94249ms/35.54679token/s。两组均1暖5、实际256/256、C1，属于小收益阶段晋级，不是最终三轮完成。

同W4AFP8-mt1017；SJ9/SJ10各8卡，AttentionTP8/专家EP8/sharedTP8，target q4 Graph、proposal3、D静态激活；D额外FULL_DRAFT_GRAPH=1。实际完整配置为本目录 full_draft_sg438_sj9_role_config.json 与 full_draft_sg438_sj10_role_config.json。

源码9046ef0292aa7bff88d2c5bcf248714e278d9ec9；实际ELF `{sha}`，`/expt/artifacts/native_sg438_runtime_timing_v1/bin/xllm`。源码清单、运行库和完整证据见[实验49结论](../02_实验记录/实验_049/05_结论与下一步.md)、[实际快照](当前阶段最佳.json)。

已通过基础题、反向OFF、第二次ON、独立16rank trace、2048/256单次冒烟及回短检查；模型进程均正常退出并释放GPU。目标尚未达到，正式长输入/最终同配置三轮/同窗口SG均未完成。仅通过基础功能检查，未完成模型精度验收。
'''
for n in ['当前累计最佳.md','当前最佳.md']:(B/n).write_text(short)
(B/'README.md').write_text('# 累计最佳入口\n\n当前为E49完整draft Graph，见[当前累计最佳](当前累计最佳.md)与[实际配置及指纹](当前阶段最佳.json)。历史E34/E39保留；仅阶段晋级，未完成最终验收。\n')
s=json.loads((C/'任务状态.json').read_text());s['previous_phase_best_new_run']=s.get('current_best_new_run');s['previous_phase_champion_metrics']=s.get('phase_champion_metrics');s.update(updated_at=now,status='E49_PROMOTED_E50_NATIVE_MICRO_RUNNING',current_best_new_run=snapshot['run'],phase_champion_metrics=confirm,current_experiment='E50 V4 SIMT Gate GEMV isolated real-weight checks',next_action='Poll existing E50 single-card checks; no restart. E49 promoted after ON/OFF/ON + long + postlong + all16 normal exit. Final targets unmet.');s['e49'].update(status='PHASE_PROMOTED_NOT_FINAL',confirm_all16_normal_exit=True,confirm_long_pass=True,post_long_pass=True,confirm_archive_sha256='a017caf298cec07c587fe6a318378dc9af14c1ed467c68ba178a8231f25c3158',phase_promotion=snapshot);s['e50']={'status':'GPU_CHECKS_RUNNING','run':'/expt/diagnostics/e50_native_gate_gemv_gpu_v1','pid':18010,'start_ticks':'524966729','container':'xiaopang1117_v41_sg438_decode_sj10','module_sha256':'face97caf02f25aaa95c916ee45c1adad7493178cb30f3ce9577964c7c5ecc2c','production_change':False};s['last_goal_turn_classification']='PROGRESS';s['last_goal_turn_progress']='E49 ON/OFF/ON and long regression complete; safely promoted small full-draft gain. E50 real-weight native SIMT micro running.';(C/'任务状态.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
with (C/'恢复记录.md').open('a') as f:f.write(f'\n\n## {now} E49闭环与E50小算例\n\nE49第二ON27.20856ms/35.17582tps，长2048/256冒烟及回短通过，16rank正常exit0、显存释放，阶段晋级并统一旧入口。最终目标未达。E50单卡检查PID18010/start524966729正在运行，禁止重复启动；候选只在隔离测试扩展，没有改生产模型。详见实验49结论与任务状态。\n')
with (C/'候选队列.md').open('a') as f:f.write(f'\n\n## {now} E49之后\n\nE49完整draft Graph阶段晋级，第二ON27.20856ms/35.17582tps，目标仍未达。E50优先复用V4原生SIMT Gate GEMV，与E31 MATE GEMM负候选不同；真实权重和动态Graph小测已启动，未替换模型。之后重新依当前target/LL/图内等待证据选择候选。最终验收与同窗口SG不变。\n')
for p in Path('/tmp').glob('match_e49_confirm*.py'):shutil.copy2(p,E/'01_代码与构建'/p.name)
for n in ['match_finalize_e49_local.py','match_e49_confirm_post_long_receipt.json','match_e49_confirm_release_sj9.json','match_e49_confirm_release_sj10_v2.json']:
 p=Path('/tmp')/n;shutil.copy2(p,E/'01_代码与构建'/p.name)
print(json.dumps({'promoted':snapshot['run'],'metrics':confirm,'goal_achieved':False},ensure_ascii=False))
