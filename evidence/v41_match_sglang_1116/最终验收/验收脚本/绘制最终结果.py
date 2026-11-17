# Copyright 2026 The xLLM Authors. All Rights Reserved.
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
root=Path(__file__).parent.parent
x=json.loads((root/'四轮最终验收复核.json').read_text())
s=json.loads((root/'SGLang同窗口复核.json').read_text())['metrics']
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
names=['xLLM 第1轮','xLLM 第2轮','xLLM 第3轮','SGLang B\n同窗口']
metrics=[
('TTFT（ms，越低越快）',[r['metrics']['TTFT_ms'] for r in x['rounds'][:3]]+[s['mean_TTFT_ms']],990.04),
('TPOT（ms/token，越低越快）',[r['metrics']['TPOT_ms'] for r in x['rounds'][:3]]+[s['mean_TPOT_ms_per_token']],21.57),
('端到端输出（token/s，越高越快）',[r['metrics']['output_tokens_s'] for r in x['rounds'][:3]]+[s['output_tokens_per_latency_sum_second']],39.45)]
fig,axes=plt.subplots(1,3,figsize=(14,4.8))
for ax,(label,values,target) in zip(axes,metrics):
 bars=ax.bar(range(4),values,color=['#287f78']*3+['#63759a'],width=.6)
 ax.set_xticks(range(4),names,fontproperties=font);ax.set_title(label,fontproperties=font)
 ax.axhline(target,color='#b54436',ls='--',lw=1.2,label=f'固定目标 {target:.2f}')
 ax.legend(loc='upper left',prop=font,frameon=False)
 for b,v in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,v,f'{v:.2f}',ha='center',va='bottom',fontsize=10)
 ax.set_ylim(0,max(max(values),target)*1.19);ax.spines[['top','right']].set_visible(False)
fig.suptitle('最终正式验收：同机、同权重、实际256/256、并发1',fontproperties=font,fontsize=16)
fig.text(.5,.025,'每组1暖机＋5正式；xLLM每轮独立重启；profiler关闭。仅通过基础功能检查，未完成模型精度验收。',ha='center',fontproperties=font,fontsize=10)
fig.tight_layout(rect=(0,.07,1,.91));fig.savefig(root/'最终正式性能对比.png',dpi=170);plt.close(fig)

