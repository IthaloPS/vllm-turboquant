#!/usr/bin/env python3
"""
TurboQuant test for local environment.
2x RTX 3090 + Qwen3.5-9B (dense model) with TP=1.

Runs two phases in separate subprocesses:
  Phase 1 - Baseline vLLM (bf16 KV)
  Phase 2 - TurboQuant (3-bit keys, 2-bit values)

Prints side-by-side comparison of VRAM usage and output quality.
"""
import os, sys, subprocess, json

MODEL = os.environ.get("MODEL", "/data/models/huggingface/Qwen__Qwen3.5-9B")
GPU_MEM = float(os.environ.get("GPU_MEM", "0.90"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "32768"))
GPUS = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
PYTHON = sys.executable

PROMPT = "Explain how KV cache compression works in large language model inference. Be detailed."
QUALITY_PROMPT = (
    "Answer precisely: "
    "1) Capital of France? "
    "2) What is 17 * 23? "
    "3) Who wrote Romeo and Juliet? "
    "4) Chemical formula for water? "
    "5) What year did WWII end?"
)


def run_phase(name, script):
    path = f"/tmp/tq_test_{name}.py"
    with open(path, "w") as f:
        f.write(script)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = GPUS
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    print(f"  Running {name}... (this may take 2-3 min)", flush=True)
    r = subprocess.run([PYTHON, path], capture_output=True, text=True, env=env, timeout=600)
    if r.returncode != 0:
        print(f"=== {name} FAILED (exit {r.returncode}) ===")
        for line in r.stderr.split("\n"):
            if any(k in line for k in ["Error", "error", "Traceback", "Exception"]):
                print(f"  {line.strip()}")
        print("--- last 20 lines of stderr ---")
        for line in r.stderr.strip().split("\n")[-20:]:
            print(f"  {line}")
        return None
    for line in reversed(r.stdout.strip().split("\n")):
        try:
            return json.loads(line)
        except Exception:
            continue
    print(f"  Warning: no JSON found in {name} output")
    print(r.stdout[-500:])
    return None


BASELINE_SCRIPT = f'''
import os, json, subprocess, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

def main():
    import time
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="{MODEL}",
        dtype="bfloat16",
        gpu_memory_utilization={GPU_MEM},
        max_model_len={MAX_MODEL_LEN},
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
    )
    blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks

    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    vram_load = [int(l.split(",")[1].strip()) for l in r.stdout.strip().split("\\n") if l.strip()]

    t0 = time.perf_counter()
    out = llm.generate(
        ["{PROMPT}"],
        SamplingParams(temperature=0, max_tokens=256))
    elapsed = time.perf_counter() - t0
    toks = len(out[0].outputs[0].token_ids)
    text = out[0].outputs[0].text[:300]

    qout = llm.generate(
        ["{QUALITY_PROMPT}"],
        SamplingParams(temperature=0, max_tokens=256))
    quality = qout[0].outputs[0].text[:400]

    r2 = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    vram_gen = [int(l.split(",")[1].strip()) for l in r2.stdout.strip().split("\\n") if l.strip()]

    print(json.dumps({{
        "blocks": blocks,
        "vram_load": vram_load,
        "vram_gen": vram_gen,
        "toks": toks,
        "elapsed": round(elapsed, 2),
        "tps": round(toks / max(elapsed, 0.01), 1),
        "text": text,
        "quality": quality,
    }}))

if __name__ == "__main__":
    main()
'''

TQ_SCRIPT = f'''
import os, json, subprocess, time
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

def main():
    import time
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="{MODEL}",
        dtype="bfloat16",
        gpu_memory_utilization={GPU_MEM},
        max_model_len={MAX_MODEL_LEN},
        tensor_parallel_size=1,
        trust_remote_code=True,
        max_num_seqs=1,
        enable_prefix_caching=False,
    )
    blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks

    engine = llm.llm_engine
    core = getattr(engine, "engine_core", engine)
    inner = getattr(core, "engine_core", core)
    executor = inner.model_executor

    def _install(worker):
        from turboquant.vllm_attn_backend import install_turboquant_hooks, MODE_ACTIVE
        states = install_turboquant_hooks(
            worker.model_runner,
            key_bits=3,
            value_bits=2,
            buffer_size=128,
            mode=MODE_ACTIVE,
        )
        return len(states)

    hooks_result = executor.collective_rpc(_install)
    hooks = hooks_result[0] if isinstance(hooks_result, list) else hooks_result
    print(f"[TurboQuant] Installed hooks on {{hooks}} layers", flush=True)

    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    vram_load = [int(l.split(",")[1].strip()) for l in r.stdout.strip().split("\\n") if l.strip()]

    t0 = time.perf_counter()
    out = llm.generate(
        ["{PROMPT}"],
        SamplingParams(temperature=0, max_tokens=256))
    elapsed = time.perf_counter() - t0
    toks = len(out[0].outputs[0].token_ids)
    text = out[0].outputs[0].text[:300]

    # Reset TQ state between generations
    def _reset(worker):
        tq_states = getattr(worker.model_runner, "_tq_states", {{}})
        for s in tq_states.values():
            s.reset()
        return len(tq_states)
    executor.collective_rpc(_reset)

    qout = llm.generate(
        ["{QUALITY_PROMPT}"],
        SamplingParams(temperature=0, max_tokens=256))
    quality = qout[0].outputs[0].text[:400]

    r2 = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    vram_gen = [int(l.split(",")[1].strip()) for l in r2.stdout.strip().split("\\n") if l.strip()]

    # Free paged KV cache
    def _free(worker):
        from turboquant.vllm_attn_backend import free_kv_cache
        return free_kv_cache(worker.model_runner)
    freed = executor.collective_rpc(_free)

    r3 = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    vram_freed = [int(l.split(",")[1].strip()) for l in r3.stdout.strip().split("\\n") if l.strip()]

    freed_bytes = freed[0] if isinstance(freed, list) else freed

    print(json.dumps({{
        "blocks": blocks,
        "hooks": hooks,
        "vram_load": vram_load,
        "vram_gen": vram_gen,
        "vram_freed": vram_freed,
        "freed_bytes": freed_bytes,
        "toks": toks,
        "elapsed": round(elapsed, 2),
        "tps": round(toks / max(elapsed, 0.01), 1),
        "text": text,
        "quality": quality,
    }}))

if __name__ == "__main__":
    main()
'''


def main():
    print("=" * 70)
    print("  TurboQuant Local Test")
    print(f"  Model:  {MODEL}")
    print(f"  GPUs:   {GPUS}")
    print(f"  GPU mem utilization: {GPU_MEM}")
    print(f"  Max model len: {MAX_MODEL_LEN}")
    print("=" * 70)
    print()

    print(">>> Phase 1: Baseline (vanilla vLLM, bf16 KV)...")
    bl = run_phase("baseline", BASELINE_SCRIPT)
    if bl is None:
        print("Baseline failed. Aborting.")
        return

    print(f"  Done. {bl['toks']} tokens in {bl['elapsed']}s ({bl['tps']} tok/s)")
    print()

    print(">>> Phase 2: TurboQuant (3-bit keys, 2-bit values)...")
    tq = run_phase("tq", TQ_SCRIPT)
    if tq is None:
        print("TurboQuant phase failed. Aborting.")
        return

    print(f"  Done. {tq['toks']} tokens in {tq['elapsed']}s ({tq['tps']} tok/s)")
    print()

    bl_vram = bl["vram_gen"][0]
    tq_vram_gen = tq["vram_gen"][0]
    tq_vram_freed = tq["vram_freed"][0]
    freed_mb = tq["freed_bytes"] / 1e6
    saved_mb = bl_vram - tq_vram_freed

    # KV cache capacity estimate
    block_size = 16
    bl_tokens = bl["blocks"] * block_size
    tq_ratio = tq["tps"] / max(bl["tps"], 0.01)

    print()
    print("=" * 70)
    print("  RESULTS")
    print()
    print(f"  {'Metric':<35} {'Baseline':>12} {'TurboQuant':>12}")
    print(f"  {'-'*35} {'-'*12} {'-'*12}")
    print(f"  {'KV cache blocks':<35} {bl['blocks']:>12,} {tq['blocks']:>12,}")
    print(f"  {'Max token capacity':<35} {bl_tokens:>12,} {bl_tokens:>12,} (same alloc)")
    print(f"  {'TQ hooks installed':<35} {'—':>12} {tq['hooks']:>12}")
    print(f"  {'VRAM after load (GPU 0)':<35} {bl['vram_load'][0]:>11} MB {tq['vram_load'][0]:>11} MB")
    print(f"  {'VRAM after generation (GPU 0)':<35} {bl_vram:>11} MB {tq_vram_gen:>11} MB")
    print(f"  {'VRAM after free_kv_cache (GPU 0)':<35} {'—':>12} {tq_vram_freed:>11} MB")
    print(f"  {'KV tensors freed':<35} {'—':>12} {freed_mb:>10.0f} MB")
    print(f"  {'VRAM saved vs baseline':<35} {'—':>12} {saved_mb:>11} MB")
    print(f"  {'Decode tok/s':<35} {bl['tps']:>12} {tq['tps']:>12}")
    print(f"  {'Throughput ratio':<35} {'1.00x':>12} {tq_ratio:>11.2f}x")
    print()
    print(f"  QUALITY CHECK (5 questions)")
    print(f"  Baseline: {bl['quality'][:200]}")
    print(f"  TQ:       {tq['quality'][:200]}")
    print()
    print(f"  GENERATION SAMPLE")
    print(f"  Baseline: {bl['text'][:150]}")
    print(f"  TQ:       {tq['text'][:150]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
