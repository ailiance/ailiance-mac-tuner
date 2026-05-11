#!/usr/bin/env python3
"""Train ailiance LoRA adapters using HF-traced data via mlx_lm_fork.

Trains sequentially: Devstral (python) → EuroLLM (chat-fr) → Apertus (math)
Uses only HF-traceable datasets for EU AI Act compliance.

Usage:
    cd ~/ailiance-mac-tuner
    .venv/bin/python scripts/train_eu_kiki_hf_batch.py
"""
import mlx.core as mx
mx.set_memory_limit(480 * 1024**3)
mx.set_cache_limit(32 * 1024**3)

import os
import sys
import time
import yaml
import shutil
from pathlib import Path

sys.path.insert(0, "/Users/clems/ailiance-mac-tuner/lib")

PROJECT = Path(__file__).parent.parent
AILIANCE = Path.home() / "ailiance"
HF_DATA = AILIANCE / "data" / "hf-traced"

JOBS = [
    {
        "name": "devstral-python-hf",
        "model": str(PROJECT / "models" / "Devstral-Small-2-24B-Instruct-2512"),
        "data": str(HF_DATA / "python"),
        "output": str(PROJECT / "output" / "ailiance-hf" / "devstral-python"),
        "adapter_dest": "devstral/python",
        "grad_accum": 4,
        "max_seq": 2048,
        "iters": 500,
    },
    {
        "name": "eurollm-chatfr-hf",
        "model": str(PROJECT / "models" / "EuroLLM-22B-Instruct-2512"),
        "data": str(HF_DATA / "chat-fr"),
        "output": str(PROJECT / "output" / "ailiance-hf" / "eurollm-chat-fr"),
        "adapter_dest": "eurollm/chat-fr",
        "grad_accum": 4,
        "max_seq": 2048,
        "iters": 500,
    },
    {
        "name": "apertus-math-hf",
        "model": str(PROJECT / "models" / "Apertus-70B-Instruct-2509"),
        "data": str(HF_DATA / "math-reasoning"),
        "output": str(PROJECT / "output" / "ailiance-hf" / "apertus-math"),
        "adapter_dest": "apertus/math",
        "grad_accum": 8,
        "max_seq": 1024,
        "iters": 500,
    },
]


def train_one(job: dict) -> bool:
    name = job["name"]
    output = job["output"]
    data = job["data"]

    # Check data exists
    train_file = Path(data) / "train.jsonl"
    if not train_file.exists():
        print(f"  SKIP {name}: no training data at {train_file}")
        return False

    n_train = sum(1 for _ in open(train_file))
    print(f"\n{'='*60}")
    print(f"TRAINING: {name}")
    print(f"  Model: {job['model'].split('/')[-1]}")
    print(f"  Data: {data} ({n_train} examples)")
    print(f"  Output: {output}")
    print(f"{'='*60}")

    os.makedirs(output, exist_ok=True)

    config = {
        "model": job["model"],
        "fine_tune_type": "lora",
        "lora_parameters": {"rank": 16, "alpha": 32, "dropout": 0.05, "scale": 2.0},
        "num_layers": -1,
        "learning_rate": 1e-5,
        "batch_size": 1,
        "grad_accumulation_steps": job["grad_accum"],
        "iters": min(job["iters"], n_train),
        "max_seq_length": job["max_seq"],
        "grad_checkpoint": True,
        "save_every": 200,
        "steps_per_report": 10,
        "steps_per_eval": 200,
        "val_batches": 5,
        "train": True,
        "seed": 42,
    }

    config_path = Path(output) / "train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    t0 = time.time()
    from mlx_lm_fork.lora import main as lora_main
    old_argv = sys.argv
    sys.argv = ["lora", "-c", str(config_path), "--data", data, "--adapter-path", output]

    try:
        lora_main()
    except KeyboardInterrupt:
        print(f"\n  {name}: interrupted (checkpoint saved)")
    except Exception as e:
        print(f"\n  {name}: ERROR — {e}")
        return False
    finally:
        sys.argv = old_argv

    elapsed = time.time() - t0
    print(f"\n  {name}: done in {elapsed/60:.1f} min")

    # Copy adapter to ailiance
    adapter_src = Path(output) / "adapters.safetensors"
    if adapter_src.exists():
        dest = AILIANCE / "output" / "adapters" / job["adapter_dest"]
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(adapter_src), str(dest / "adapters.safetensors"))
        size_mb = adapter_src.stat().st_size / 1048576
        print(f"  Copied adapter ({size_mb:.0f} MB) → {dest}")

    return True


def main():
    print("ailiance HF-traced LoRA batch training")
    print(f"Jobs: {len(JOBS)}")

    results = []
    for job in JOBS:
        ok = train_one(job)
        results.append((job["name"], ok))

    print(f"\n{'='*60}")
    print("BATCH RESULTS:")
    for name, ok in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
