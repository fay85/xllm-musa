"""Prepare reviewable evidence; never commits or pushes automatically."""
from pathlib import Path
import shutil,json,hashlib,re
b=Path(__file__).resolve().parents[1];repo=b.parents[1]/'20261105跑通PD分离/代码/xllm-musa';dest=repo/'docs/experiments/20261110_deepseek_v41_mhc_1p1d';dest.mkdir(parents=True,exist_ok=True)
for name in ('今日工作总结.md','今晚待运行的goal.md','给家明的阶段性汇报.md'):
 shutil.copy2(b.parent/name,dest/name)
if (b.parent/'commit_message.txt').exists():shutil.copy2(b.parent/'commit_message.txt',dest/'commit_message.txt')
for name in ('mHC接入与数值问题说明.md','代码复核.md','固定前缀探针与V2协议缺口.md','逐项算子完整模型对照.md'):
 out=dest/'阶段一二'/name;out.parent.mkdir(exist_ok=True);shutil.copy2(b/name,out)
for directory in ('dev6_mhc','sj9_operator_preflight'):
 target=dest/'阶段一二/证据'/directory;target.mkdir(parents=True,exist_ok=True)
 for p in (b/'证据'/directory).iterdir():
  if p.is_file():shutil.copy2(p,target/p.name)
for p in (b/'证据').glob('*.json'):
 if p.stat().st_size>1000000:continue
 if p.name in ('latest_observation.json','matrix_status.json'):continue
 target=dest/'阶段一二/证据'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
for p in (b/'证据').glob('*.csv'):shutil.copy2(p,dest/'阶段一二/证据'/p.name)
for p in (b/'证据').glob('*.log'):shutil.copy2(p,dest/'阶段一二/证据'/p.name)
for p in (b/'证据').glob('*.patch'):shutil.copy2(p,dest/'阶段一二/证据'/p.name)
# State and execution logs are snapshots, not a claim that all runs completed.
for p in list((b/'脚本').glob('*.py'))+list((b/'脚本').glob('*.sh')):
 target=dest/'阶段一二/脚本'/p.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
(dest/'README.md').write_text('''# V4.1：1017 双机基线与三项 mHC 原语接入

本记录接续 [共享 SwiGLU 的实现与单卡验证](../20261110_deepseek_v41_ops/README.md)。查看 [今日工作总结](今日工作总结.md) 中的实际验收状态与完整模型性能；只有状态为 PASS 的原始数据库才能作为性能结果。

- [mHC 实现、数值失败及处理](阶段一二/mHC接入与数值问题说明.md)
- [逐项完整模型对照](阶段一二/逐项算子完整模型对照.md)
- [固定前缀探针的服务协议缺口](阶段一二/固定前缀探针与V2协议缺口.md)
- [后续 Python 和 C++ 工作](今晚待运行的goal.md)
- [给家明的阶段性说明](给家明的阶段性汇报.md)

`阶段一二/证据/dev6_mhc` 保存编译、失败、数值诊断、真实参数检查与小模型结果；`sj9_operator_preflight` 是实际 SJ 硬件上的复验。`.pt` 是我方生成的测试/诊断张量，没有提交 1017 模型权重子集。

[当次编译脚本](阶段一二/脚本/build_stage2.sh) 的执行位置是 dev6 的我方构建容器，保留了当次容器内路径。它重新构建 xllm 和测试扩展，并复用镜像内的 Rust tokenizer/safetensors 及 vcpkg 依赖；控制服务继续使用镜像预置产物。源码、构建与实际运行产物的 SHA 分别保留，不能把这次增量构建称为全量依赖从源码重建。

完整实验原始文件位于 NFS `/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops/`。完成归档后，`raw_runs/` 保存完整实验压缩包，解压得到原始 SQLite、HTML、逐请求记录、rank 日志、配置与基本问答。校验值见该目录 manifest。失败实验保留 FAIL 标记，不混入成功结果。

本次执行脚本依赖 AI Book 工作目录里的 `sj9_12_登陆工具/sj_access.py` 取得已配置的 SSH 访问；该凭据工具不随报告发布。数值回归脚本位于仓库 `tools/musa_deepseek_v41/`，可使用自己的环境路径运行。报告中的三项参考调研等外部归档路径指 AI Book 原任务目录；不是此仓库内的安装依赖。

本轮仅向个人开发分支提交，不创建 MR/PR。完整模型仍是 Python forward 调用 C++ 绑定及编译 GPU kernel，未宣称改为原生 C++ V4.1 模型。
''')
findings=[]
pattern=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|sk-[A-Za-z0-9_-]{25,})')
for p in dest.rglob('*'):
 if p.is_file() and p.suffix!='.pt' and pattern.search(p.read_bytes()):findings.append(str(p.relative_to(dest)))
assert not findings,findings
print(json.dumps({'prepared':str(dest),'files':sum(p.is_file() for p in dest.rglob('*')),'bytes':sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),'credential_pattern_hits':findings},ensure_ascii=False))
