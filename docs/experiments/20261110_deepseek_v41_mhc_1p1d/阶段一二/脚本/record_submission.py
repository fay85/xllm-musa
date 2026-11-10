"""After a successful push, record the verified commit outside its own contents."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess

base = Path(__file__).resolve().parents[1]
parent = base.parent
repo = base.parents[1] / '20261105跑通PD分离/代码/xllm-musa'

def git(*args: str) -> str:
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()

branch = git('branch', '--show-current')
assert branch == 'xiaopangDeepseeekV41Flash', branch
sha = git('rev-parse', 'HEAD')
remote = git('ls-remote', 'origin', 'refs/heads/' + branch)
assert remote.split()[0] == sha, remote
release = json.loads((base / '证据/final_resource_release.json').read_text())
assert release['status'] == 'OWN_MODEL_AND_CONTROL_PROCESSES_EXITED_NORMALLY'
manifest = json.loads((base / '证据/stage2_source_manifest.json').read_text())
production = {}
for name, expected in manifest['changed_files'].items():
    if name.startswith('xllm/'):
        actual = hashlib.sha256((repo / name).read_bytes()).hexdigest()
        assert actual == expected, name
        production[name] = actual
url = 'https://github.com/fay85/xllm-musa/commit/' + sha
branch_url = 'https://github.com/fay85/xllm-musa/tree/' + branch
record = {
    'at': datetime.datetime.now().astimezone().isoformat(),
    'commit': sha, 'commit_url': url, 'branch': branch, 'branch_url': branch_url,
    'remote_head_verified': remote,
    'compiled_source_parent': manifest['parent_commit'],
    'compiled_source_archive_sha256': manifest['source_archive_sha256'],
    'production_files_match_tested_build': production,
    'binary_sha256': '862d3ab3a8a1eab99326c353782af0b7a170041e53174063735a27051ae0a5af',
    'accuracy_boundary': '79 operator checks and tiny model/basic questions passed; full-model same-prefix numerical parity remains unverified; opt-in fusion, reference remains default.',
    'merge_request_created': False,
}
(base / '证据/final_submission.json').write_text(json.dumps(record, ensure_ascii=False, indent=2))
message = git('log', '-1', '--format=%B')
assert message.strip() == (parent / 'commit_message.txt').read_text().strip()
report = f'''# 本轮最终提交与交付记录

记录时间：{record['at']}。

- 共同仓库个人分支：[{branch}]({branch_url})。
- 本次提交：[{sha}]({url})。
- 已重新读取远端 refs/heads/{branch}，SHA 与本地提交一致。
- 只推送个人分支，没有创建 MR/PR。
- 本轮五种配置的两组 1+5 测试全部完成；原语/小模型和基础问答通过，真实整模型同前缀数值验证仍有缺口，不能称为完整精度验收。
- 本次构建的三个生产文件 SHA 与最终提交一致。二进制 SHA256 为 `{record['binary_sha256']}`；验证脚本后续扩展与生产代码分开记录。
- 两机本轮模型及控制进程均正常退出。每卡显存终态、容器名和状态见 `阶段一二/证据/final_resource_release.json`。

报告入口：[今日工作总结](今日工作总结.md)、[给家明的说明](给家明的阶段性汇报.md)、[后续计划](今晚待运行的goal.md)。

NFS 根目录：`/data/nfs_shared/xiaopang/deepseek_v41_flash_1p1d/20261110_1103_ops/`。
报告镜像：该目录 `04_reports/`。原始结果：`03_benchmarks/` 与 `runs/`。
仓库内归档：`docs/experiments/20261110_deepseek_v41_mhc_1p1d/raw_runs/`。

本文件在提交完成并核对远端后生成，保存在 AI Book 与 NFS；不会为了把提交自身的 SHA 写进该提交内容而循环改写 commit。

## 实际 commit message（可复制给家明）

```text
{message}
```
'''
(parent / '最终提交与交付记录.md').write_text(report)
p = parent / '给家明的阶段性汇报.md'
s = p.read_text().replace('本轮 mHC 代码和全部实验记录纳入本次提交，最终提交网址另存于本地及 NFS 的“最终提交与交付记录.md”。', f'本轮 mHC 代码和全部实验记录已提交并推送到 [{sha[:8]}]({url})，远端分支 SHA 已核对。')
p.write_text(s)
p = parent / '今日工作总结.md'
s = p.read_text() + f'\n## 9. 最终提交确认\n\n已推送至共同仓库个人分支，提交 [{sha}]({url})；远端 SHA 已核对一致。提交说明见 `commit_message.txt`，交付对应关系见 `最终提交与交付记录.md`。未创建 MR/PR。\n'
p.write_text(s)
p = base / '执行接力.json'
d = json.loads(p.read_text())
d.update(at=record['at'], controller_pid=None, active_controller_session=None,
         controller='Finished; do not restart completed matrix', current_run=None,
         latest_completed_run='stage2_all', E_status='Both 1+5 groups validated, all ranks/control normal0, archived.',
         next='This task handed off with reports and verified pushed commit. See 今晚待运行的goal.md for subsequent work.',
         uncompleted='Full-model same-prefix numerical parity and remaining high-performance modules explicitly deferred; no active GPU experiment.',
         final_commit=sha, final_commit_url=url)
p.write_text(json.dumps(d, ensure_ascii=False, indent=2))
print(json.dumps(record, ensure_ascii=False))
