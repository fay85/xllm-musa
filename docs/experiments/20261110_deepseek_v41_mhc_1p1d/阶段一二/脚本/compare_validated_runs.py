"""Compare only fully validated runs; preserve text agreement and variability."""
from pathlib import Path
import json,math,datetime
b=Path(__file__).resolve().parents[1];e=b/'证据'
selection=json.loads((e/'shared_swiglu_selection.json').read_text()) if (e/'shared_swiglu_selection.json').exists() else None
pairs=[('stage1_ref','stage1_swiglu_d_v2','D 共享 SwiGLU')]
if selection:
 parent='stage1_swiglu_d_v2' if selection['D_shared_swiglu']=='v4_fused' else 'stage1_ref'
 pairs.extend([(parent,'stage2_post','mHC post'),('stage2_post','stage2_post_pre','mHC pre-combine'),('stage2_post_pre','stage2_all','mHC metadata/Sinkhorn')])
rows=[]
for old,new,label in pairs:
 p=e/f'{old}_validated_summary.json';q=e/f'{new}_validated_summary.json'
 if not p.exists() or not q.exists():continue
 a,z=json.loads(p.read_text()),json.loads(q.read_text());assert a['status']==z['status']=='PASS'
 for x,y in zip(a['results'],z['results']):
  assert x['input_tokens']==y['input_tokens'] and x['output_tokens']==y['output_tokens']==256 and x['repeats']==y['repeats']==5
  m,n=x['metrics']['tpot_ms'],y['metrics']['tpot_ms'];delta=m['mean']-n['mean'];two_se=2*math.sqrt((m['sample_stddev']**2+n['sample_stddev']**2)/5)
  rows.append({'identical_request_bodies':len(x.get('distinct_request_body_hashes',[]))==len(y.get('distinct_request_body_hashes',[]))==1 and x['distinct_request_body_hashes']==y['distinct_request_body_hashes'],'operator':label,'before':old,'after':new,'input_tokens':x['input_tokens'],'output_tokens':256,'before_tpot_ms':m['mean'],'after_tpot_ms':n['mean'],'tpot_reduction_ms':delta,'tpot_reduction_pct':100*delta/m['mean'],'ttft_reduction_pct':100*(1-y['metrics']['ttft_s']['mean']/x['metrics']['ttft_s']['mean']),'output_throughput_increase_pct':100*(y['output_tokens_over_test_duration']/x['output_tokens_over_test_duration']-1),'two_combined_standard_errors_ms':two_se,'practical_noise_threshold_ms':max(two_se,.002*m['mean']),'exceeds_practical_improvement_threshold':delta>max(two_se,.002*m['mean']),'all_repeated_output_text_identical_and_equal':len(x['distinct_response_text_hashes'])==len(y['distinct_response_text_hashes'])==1 and x['distinct_response_text_hashes']==y['distinct_response_text_hashes']})
result={'at':datetime.datetime.now().astimezone().isoformat(),'comparisons':rows,'interpretation':'Differences use sequential batches, five samples per group. The noise threshold is a practical selection rule, not a controlled causal proof or a general significance claim. Output text agreement is not full-vocabulary logits equivalence.','shared_swiglu_selection':selection}
(e/'operator_full_model_comparisons.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
lines=['# 逐项算子接入的完整模型性能对照','','只采用已通过原始请求与服务日志验收的 1 次暖机 + 5 次正式结果。缺项表示尚未完成。','', '| 本次接入 | 输入/输出 | TPOT 前→后 ms/token | TPOT 耗时下降 | 输出吞吐提升 | 两边各 5 次输出文本相同 |','|---|---|---:|---:|---:|---|']
for r in rows:lines.append(f"| {r['operator']} | {r['input_tokens']}/256 | {r['before_tpot_ms']:.2f} → {r['after_tpot_ms']:.2f} | {r['tpot_reduction_pct']:+.2f}% | {r['output_throughput_increase_pct']:+.2f}% | {'是' if r['all_repeated_output_text_identical_and_equal'] else '否，需单独检查'} |")
lines+=['','负的耗时下降表示变慢；负的吞吐提升表示吞吐下降。顺序采样的差异仍可能受到环境波动影响；若输出文本不同，不能简单归因全部性能差异来自 kernel。每个算子使用哪一个前置配置、样本标准差及预设噪声阈值详见 `证据/operator_full_model_comparisons.json`。','', '本轮 SwiGLU 用旧二进制，mHC 用新增绑定后的二进制。mHC post 与前置配置比较同时跨越了这次二进制构建和静态后端选择代码；不能称为硬件计数级的纯单 kernel 因果隔离。后续 pre 与 metadata 使用同一个 mHC 二进制。']
(b/'逐项算子完整模型对照.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,ensure_ascii=False))
