"""Offline eight-rank probe: alternating BF16 and FP32 on one native custom communicator."""
from __future__ import annotations
import argparse
import datetime
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def _worker(rank: int, host: str, port: int, out: Path) -> None:
    import torch
    import torch_musa
    import torch.distributed as dist
    import musa_python_kernel_test_module
    torch.set_num_threads(1)
    torch.musa.set_device(rank)
    store=dist.TCPStore(host,port,8,rank==0,datetime.timedelta(seconds=90))
    dist.init_process_group('mccl',store=dist.PrefixStore('mccl/',store),rank=rank,world_size=8,timeout=datetime.timedelta(seconds=30))
    ops=torch.ops._C_custom_ar
    local_allocations=[]
    def share(size: int, label: str) -> list[int]:
        pointer,handle=ops.allocate_shared_buffer_and_handle(size)
        local_allocations.append(pointer)
        keys=[label+'/'+str(i) for i in range(8)]
        store.set(keys[rank],handle.cpu().numpy().tobytes())
        store.wait(keys)
        return [pointer if i==rank else ops.open_mem_handle(torch.tensor(list(store.get(keys[i])),dtype=torch.uint8)) for i in range(8)]
    max_bytes=8*1024*1024
    meta=share(ops.meta_size()+max_bytes,'meta')
    buffers=share(max_bytes,'buffer')
    rank_data=torch.empty(8*1024*1024,device='musa:'+str(rank),dtype=torch.uint8)
    handle=ops.init_custom_ar(meta,rank_data,rank,True)
    ops.set_algorithm(handle,1)
    ops.register_buffer(handle,buffers)
    def vote(label: str, valid: bool) -> None:
        keys=[label+'/'+str(i) for i in range(8)];store.set(keys[rank],b'1' if valid else b'0');store.wait(keys)
        if any(store.get(k)!=b'1' for k in keys):raise AssertionError('Synchronized numerical/capability failure: '+label)
    vote('initialized',True)
    result={'rank':rank,'device':rank,'world_size':8,'group':'tp_group algorithm1 / native dispatcher','model_width':5120,'results':[],'numeric':[],'interpretation':'All-ready API/stream latency probe; Python enqueue overhead remains, not pure wire time and not full-model performance.'}
    with torch.inference_mode():
        base=(torch.arange(5120,dtype=torch.float32).remainder(31)-15)/64
        source=(base+rank/16).to(torch.bfloat16).to('musa:'+str(rank))
        expected=(8*base+28/16)*4.5
        result['purpose']='100 dependent BF16 custom TP -> FP32 arithmetic -> custom EP repetitions, current stream, no per-call device synchronize'
        dist.barrier();torch.musa.synchronize()
        begin=time.perf_counter()
        for iteration in range(100):
            attention=torch.empty_like(source)
            ops.all_reduce(handle,source,attention,buffers[rank],max_bytes)
            routed=attention.float()*(rank+1)/8
            reduced_routed=torch.empty_like(routed)
            ops.all_reduce(handle,routed,reduced_routed,buffers[rank],max_bytes)
            routed=reduced_routed
        torch.musa.synchronize()
        diff=(routed.cpu()-expected).abs().max().item()
        result['mixed']={'repetitions':100,'max_abs':diff,'exact':torch.equal(routed.cpu(),expected),'elapsed_s':time.perf_counter()-begin,'scope':'ordering and buffer-reuse regression, not model performance'}
        vote('mixed_exact',result['mixed']['exact'])
    vote('all_gpu_work_done',True)
    ops.dispose(handle)
    # Everyone has finished GPU work before any local IPC allocation is freed.
    vote('all_handles_disposed',True)
    for ptr in local_allocations:ops.free_shared_buffer(ptr)
    dist.destroy_process_group()
    (out/('rank'+str(rank)+'.json')).write_text(json.dumps(result,indent=2))


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--rank',type=int);parser.add_argument('--host',required=True);parser.add_argument('--port',type=int,default=22541);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.rank is not None:
        _worker(args.rank,args.host,args.port,args.output);return
    args.output.mkdir(parents=True,exist_ok=False)
    children=[];logs=[]
    for rank in range(8):
        log=(args.output/('rank'+str(rank)+'.log')).open('wb');logs.append(log)
        children.append(subprocess.Popen([sys.executable,__file__,'--rank',str(rank),'--host',args.host,'--port',str(args.port),'--output',str(args.output)],stdout=log,stderr=subprocess.STDOUT))
    (args.output/'launch.json').write_text(json.dumps({'pids':[p.pid for p in children],'started_at':datetime.datetime.now().astimezone().isoformat(),'note':'Only this exact owned-container eight-process probe; no SIGKILL/restart.'},indent=2))
    term_sent=False;deadline=time.monotonic()+300
    while any(p.poll() is None for p in children):
        if not term_sent and (any(p.poll() not in [None,0] for p in children) or time.monotonic()>deadline):
            for p in children:
                if p.poll() is None:p.terminate()
            term_sent=True
        (args.output/'status.json').write_text(json.dumps({'pids':[p.pid for p in children],'exit_codes':[p.poll() for p in children],'TERM_sent':term_sent}))
        time.sleep(1)
    codes=[p.returncode for p in children]
    for log in logs:log.close()
    result={'exit_codes':codes,'TERM_sent':term_sent,'SIGKILL_sent':False,'finished_at':datetime.datetime.now().astimezone().isoformat()}
    (args.output/'result.json').write_text(json.dumps(result,indent=2))
    if codes!=[0]*8:raise RuntimeError(result)


if __name__=='__main__':main()
