"""Offline eight-rank probe of unchanged xLLM custom-AR native kernels."""
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
        for dtype in [torch.bfloat16,torch.float32]:
            dtype_name=str(dtype)
            base=(torch.arange(5120,dtype=torch.float32).remainder(31)-15)/64
            source=(base+rank/16).to(dtype).to('musa:'+str(rank))
            expected=(8*base+28/16).to(dtype)
            first=source.clone();native=torch.empty_like(source)
            dist.all_reduce(first);ops.all_reduce(handle,source,native,buffers[rank],max_bytes);torch.musa.synchronize()
            valid=torch.equal(first.cpu(),expected) and torch.equal(native.cpu(),expected)
            result['numeric'].append({'dtype':dtype_name,'test':'signed exact dyadic values','mccl_exact':torch.equal(first.cpu(),expected),'custom_exact':torch.equal(native.cpu(),expected)})
            vote(dtype_name+'/exact',valid)
            # The same per-rank FP32 construction is quantized to the tested
            # dtype before either reduction, and both compare to the FP32 sum.
            seeds=[]
            for peer in range(8):
                gen=torch.Generator().manual_seed(111200+peer);seeds.append(torch.randn(5120,generator=gen).to(dtype))
            oracle=torch.stack([x.float() for x in seeds]).sum(0)
            x=seeds[rank].to('musa:'+str(rank));m=x.clone();c=torch.empty_like(x)
            dist.all_reduce(m);ops.all_reduce(handle,x,c,buffers[rank],max_bytes);torch.musa.synchronize()
            errors={}
            for name,val in [('mccl',m),('custom',c)]:
                diff=val.cpu().float()-oracle;errors[name]={'max_abs':diff.abs().max().item(),'relative_rms':(diff.square().mean().sqrt()/oracle.square().mean().sqrt()).item(),'finite':bool(torch.isfinite(val).all())}
            limit=.01 if dtype==torch.bfloat16 else .000003
            valid=all(v['finite'] and v['relative_rms']<=limit for v in errors.values())
            result['numeric'].append({'dtype':dtype_name,'test':'fixed per-rank random inputs vs FP32 sum','relative_rms_limit':limit,'errors':errors})
            vote(dtype_name+'/random',valid)
            trials={'mccl':[],'custom':[]}
            # One ten-call warmup, then five independent batches of 100 calls.
            # Alternate the backend order by trial; all eight ranks agree.
            for trial in range(6):
                count=10 if trial==0 else 100
                order=['mccl','custom'] if trial%2==0 else ['custom','mccl']
                for backend in order:
                    inp=source.unsqueeze(0).expand(count,-1).clone();outputs=torch.empty_like(inp)
                    dist.barrier();torch.musa.synchronize()
                    start=torch.musa.Event(enable_timing=True);end=torch.musa.Event(enable_timing=True)
                    begin=time.perf_counter();start.record()
                    for i in range(count):
                        if backend=='mccl':dist.all_reduce(inp[i])
                        else:ops.all_reduce(handle,inp[i],outputs[i],buffers[rank],max_bytes)
                    launch_ms=(time.perf_counter()-begin)*1000
                    end.record();end.synchronize();wall_ms=(time.perf_counter()-begin)*1000
                    actual=inp[-1] if backend=='mccl' else outputs[-1]
                    good=torch.equal(actual.cpu(),expected)
                    vote(dtype_name+'/'+str(trial)+'/'+backend,good)
                    if trial:
                        trials[backend].append({'trial':trial,'operations':count,'gpu_batch_ms_per_op':start.elapsed_time(end)/count,'host_enqueue_ms_per_op':launch_ms/count,'host_wall_ms_per_op':wall_ms/count,'last_result_exact':good})
            result['results'].append({'dtype':dtype_name,'payload_bytes':source.nbytes,'trials':trials,'median_gpu_batch_ms_per_op':{k:statistics.median(v['gpu_batch_ms_per_op'] for v in rs) for k,rs in trials.items()}})
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
