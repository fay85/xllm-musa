import os, sys, time
sys.path.insert(0, "/workspace/xllm-git-master/build/lib.linux-x86_64-cpython-310")
import torch
import torch_musa
from torch.profiler import profile, ProfilerActivity

os.environ.setdefault("MUSA_VISIBLE_DEVICES", "1")

from xllm import LLM, SamplingParams

MODEL = "/workspace/model_weights/Qwen3.5-27B"

print("Loading model...", flush=True)
llm = LLM(
    model=MODEL,
    devices="musa:0",
    block_size=64,
    max_memory_utilization=0.8,
    enable_prefix_cache=False,
    enable_chunked_prefill=True,
    enable_schedule_overlap=False,
    max_tokens_per_chunk_for_prefill=8192,
    max_seqs_per_batch=8,

    enable_graph=True,
    enable_graph_mode_decode_no_padding=True,
    enable_prefill_piecewise_graph=True,
    max_tokens_for_graph_mode=8192,
    use_cpp_chat_template=True,
    disable_log_stats=True,
)
print("Model loaded.", flush=True)

short_prompt = "Hello"
long_prompt = " ".join(["word"] * 1700)

print("\n=== WARMUP ===", flush=True)
out = llm.generate(short_prompt, SamplingParams(max_tokens=8, temperature=0), use_tqdm=False)
print(f"warmup done: {out[0].outputs[0].text[:80]}", flush=True)

print("\n=== PROFILING PREFILL (ISL~2500, OSL=1) ===", flush=True)
sp_prefill = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MUSA]) as prof_prefill:
    out = llm.generate(long_prompt, sp_prefill, use_tqdm=False)
print(f"prefill done: pt={out[0].usage.prompt_tokens}", flush=True)
print("\n--- PREFILL top kernels ---", flush=True)
print(prof_prefill.key_averages().table(sort_by="musa_time_total", row_limit=30), flush=True)

print("\n=== PROFILING DECODE (C=5, OSL=20) ===", flush=True)
sp_decode = SamplingParams(max_tokens=20, temperature=0, ignore_eos=True)
prompts = [long_prompt] * 5
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.MUSA]) as prof_decode:
    out = llm.generate(prompts, sp_decode, use_tqdm=False)
print(f"decode done: {len(out)} requests", flush=True)
print("\n--- DECODE top kernels ---", flush=True)
print(prof_decode.key_averages().table(sort_by="musa_time_total", row_limit=30), flush=True)

print("\n=== DONE ===", flush=True)
llm.finish()
