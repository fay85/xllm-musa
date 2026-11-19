"""Summarize existing P rank0 diagnostic boundaries, not device busy time."""
import argparse
import datetime as dt
import json
import re
from pathlib import Path


def stages(path):
    records, chunks, active = {}, [], None
    for line in path.open(errors="replace"):
        layer = re.search(r"V41_DIAGNOSTIC layer (begin|enqueued) index=(\d+) q=(\d+) kv=(\d+)", line)
        if layer and (layer.group(1, 2) in (("begin", "0"), ("enqueued", "39"))):
            stamp = dt.datetime.strptime(line[1:25], "%Y%m%d %H:%M:%S.%f")
            shape = tuple(map(int, layer.group(3, 4)))
            if layer[1] == "begin":
                if active is not None:
                    raise ValueError("unpaired layer begin")
                active = (stamp, shape)
            else:
                if active is None or shape != active[1]:
                    raise ValueError("unpaired layer end")
                chunks.append(dict(q=shape[0], kv=shape[1],
                                   rank0_host_span_ms=1000*(stamp-active[0]).total_seconds()))
                active = None
        if "record_type: pd_prefill_stage," in line:
            rid = re.search(r"request_id: ([^,]+)", line)[1]
            prompt = int(re.search(r"prompt_tokens: (\d+)", line)[1])
            if rid in records or active or sum(c["q"] for c in chunks) != prompt:
                raise ValueError(f"chunk boundaries do not account for {rid}")
            records[rid] = dict(chunks=chunks, prompt_tokens=prompt,
                                server_ttft_ms=float(re.search(r"ttft: ([\d.]+)ms", line)[1]),
                                queue_to_batch_ms=float(re.search(r"queue_to_batch: ([\d.]+)ms", line)[1]))
            chunks = []
    if not records or chunks or active:
        raise ValueError("unfinished or missing P records")
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("prefill_rank0_log", type=Path)
    print(json.dumps(stages(ap.parse_args().prefill_rank0_log), indent=2))
