# Copyright 2026 The xLLM Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import json,statistics,collections,sys,math
W=Path('/home/xiaopang/桌面/Albert分享/20261021从xllm开始');P=W/'王家明/20261111后续推进Cpp吧/以sglang_mt工作为母本去做/20261112新的开始';E=P/'ToDoList_3';sys.path.insert(0,str(E/'05_trace'));from calibrate_sampled_steps import requests
req={n:requests(E/'04_性能/DEVICE/raw/performance',n) for n in [256,2048]};d=json.loads((E/'05_trace/DEVICE/sj10/scopes.json').read_text())['measurements'];g=collections.defaultdict(list)
for r in d:
 if r['q']!=1 or r['scope'] not in ['V41/Attention/TP_allreduce','V41/MoE/EP_allreduce']:continue
 matches=[(n,i) for n,rs in req.items() for i,x in enumerate(rs) if x['start_time']<=r['host_begin_ns']/1e9<=x['completed_time']]
 if not matches:continue
 assert len(matches)==1
 n,i=matches[0];g[(n,i,r['step'],r['layer'],r['scope'])].append(r)
rows=[]
for (n,i,step,layer,scope),rr in g.items():
 assert len(rr)==8 and len({x['pid'] for x in rr})==8,(n,i,step,layer,scope,len(rr))
 times=[x['host_begin_ns']/1e6 for x in rr];max_t=max(times);wait=[max_t-x for x in times];gpu=[x['gpu_stream_span_ms'] for x in rr];mw=statistics.mean(wait);mg=statistics.mean(gpu);varw=sum((x-mw)**2 for x in wait);varg=sum((x-mg)**2 for x in gpu);rho=sum((x-mw)*(y-mg) for x,y in zip(wait,gpu))/math.sqrt(varw*varg) if varw>0 and varg>0 else None
 rows.append({'input_tokens':n,'request_index':i,'step':step,'layer':layer,'scope':scope,'CPU_enqueue_range_ms':max_t-min(times),'GPU_span_min_ms':min(gpu),'GPU_span_mean_ms':mg,'GPU_span_max_ms':max(gpu),'CPU_earliness_vs_GPU_duration_correlation':rho,'payload_bytes':sorted({x['payload_bytes'] for x in rr}),'group_sizes':sorted({x['group_size'] for x in rr})})
out={'status':'SAME_HOST_ENQUEUE_PROXY_NOT_GPU_READY_TIME','limitations':['CPU range begins before event/collective submission; GPU backlog may differ.','Correlation is a diagnostic clue, not proof of slow-rank causality.','Event spans contain dependencies and waiting; do not call them pure wire transfer.'],'records':rows,'summary':{}}
for n in [256,2048]:
 for scope in ['V41/Attention/TP_allreduce','V41/MoE/EP_allreduce']:
  rs=[r for r in rows if r['input_tokens']==n and r['scope']==scope];out['summary'][str(n)+'/'+scope]={'groups':len(rs),'cpu_enqueue_range_median_ms':statistics.median(x['CPU_enqueue_range_ms'] for x in rs),'gpu_span_min_median_ms':statistics.median(x['GPU_span_min_ms'] for x in rs),'gpu_span_max_median_ms':statistics.median(x['GPU_span_max_ms'] for x in rs),'correlation_median':statistics.median(x['CPU_earliness_vs_GPU_duration_correlation'] for x in rs if x['CPU_earliness_vs_GPU_duration_correlation'] is not None)}
(E/'05_trace/collective_CPU到达与设备跨度.json').write_text(json.dumps(out,indent=2));print(json.dumps(out['summary'],indent=2))
