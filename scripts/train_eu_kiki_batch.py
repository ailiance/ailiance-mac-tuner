#!/usr/bin/env python3
"""Batch train all ailiance LoRA adapters sequentially.

Usage:
    cd ~/ailiance-mac-tuner
    .venv/bin/python scripts/train_eu_kiki_batch.py

Trains adapters for 3 models across all domains.
Skips domains that already have adapters.
"""
import mlx.core as mx
mx.set_memory_limit(480 * 1024**3)
mx.set_cache_limit(32 * 1024**3)

import os
import sys
import time
import yaml
from pathlib import Path

sys.path.insert(0, "/Users/clems/ailiance-mac-tuner/lib")

PROJECT = Path(__file__).parent.parent
AILIANCE = Path.home() / "ailiance"
DATA_ROOT = PROJECT / "data" / "micro-kiki"

# Model configs: (model_path, output_prefix, domains, grad_accum, max_seq, iters)
MODELS = {
    "apertus": {
        "model": str(PROJECT / "models" / "Apertus-70B-Instruct-2509"),
        "grad_accum": 8,
        "max_seq": 1024,
        "iters": 500,
        "domains": [
            "electronics", "emc", "dsp", "spice", "kicad-dsl", "kicad-pcb",
            "stm32", "platformio", "iot", "embedded", "math", "reasoning",
            "security", "music-audio", "freecad", "power", "spice-sim",
        ],
    },
    "devstral": {
        "model": str(PROJECT / "models" / "Devstral-Small-2-24B-Instruct-2512"),
        "grad_accum": 4,
        "max_seq": 2048,
        "iters": 500,
        "domains": [
            "python", "rust", "typescript", "cpp", "shell", "html-css",
            "sql", "web-backend", "web-frontend", "docker", "devops",
            "yaml-json", "llm-orch", "lua-upy",
        ],
    },
    "eurollm": {
        "model": str(PROJECT / "models" / "EuroLLM-22B-Instruct-2512"),
        "grad_accum": 4,
        "max_seq": 2048,
        "iters": 500,
        "domains": [
            "chat-fr",
        ],
    },
}


def find_data(domain: str) -> Path | None:
    """Find training data for a domain."""
    # Check if split data exists
    for subdir in [domain, domain.replace("-", "_")]:
        train = DATA_ROOT / subdir / "train.jsonl"
        if train.exists():
            return DATA_ROOT / subdir
    # Check classified
    classified = DATA_ROOT / "classified" / f"{domain}.jsonl"
    if classified.exists():
        # Create split
        import json
        import random
        random.seed(42)
        out = DATA_ROOT / domain
        out.mkdir(exist_ok=True)
        with open(classified) as f:
            rows = [json.loads(line) for line in f]
        random.shuffle(rows)
        cut = max(1, int(len(rows) * 0.95))
        with open(out / "train.jsonl", "w") as f:
            for r in rows[:cut]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(out / "valid.jsonl", "w") as f:
            for r in rows[cut:]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  Split {domain}: {cut} train / {len(rows) - cut} valid")
        return out
    return None


def adapter_exists(model_name: str, domain: str) -> bool:
    path = AILIANCE / "output" / "adapters" / model_name / domain / "adapters.safetensors"
    return path.exists()


def train_one(model_name: str, model_cfg: dict, domain: str):
    """Train one LoRA adapter."""
    output_dir = str(PROJECT / "output" / "ailiance" / f"{model_name}-{domain}")
    os.makedirs(output_dir, exist_ok=True)

    data_path = find_data(domain)
    if data_path is None:
        print(f"  SKIP {domain}: no training data found")
        return False

    n_train = sum(1 for _ in open(data_path / "train.jsonl"))
    if n_train < 10:
        print(f"  SKIP {domain}: only {n_train} examples (need >= 10)")
        return False

    config = {
        "model": model_cfg["model"],
        "fine_tune_type": "lora",
        "lora_parameters": {"rank": 16, "alpha": 32, "dropout": 0.05, "scale": 2.0},
        "num_layers": -1,
        "learning_rate": 1e-5,
        "batch_size": 1,
        "grad_accumulation_steps": model_cfg["grad_accum"],
        "iters": min(model_cfg["iters"], n_train),
        "max_seq_length": model_cfg["max_seq"],
        "grad_checkpoint": True,
        "save_every": 200,
        "steps_per_report": 10,
        "steps_per_eval": 200,
        "val_batches": 5,
        "train": True,
        "seed": 42,
    }

    config_path = Path(output_dir) / "train_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"  Training {model_name}/{domain} ({n_train} examples, {config['iters']} iters)...")
    t0 = time.time()

    from mlx_lm_fork.lora import main as lora_main
    old_argv = sys.argv
    sys.argv = ["lora", "-c", str(config_path),
                "--data", str(data_path),
                "--adapter-path", output_dir]
    try:
        lora_main()
    except Exception as e:
        print(f"  ERROR {domain}: {e}")
        return False
    finally:
        sys.argv = old_argv

    elapsed = time.time() - t0
    print(f"  DONE {domain} in {elapsed / 60:.1f} min")

    # Copy adapter to ailiance
    adapter_src = Path(output_dir) / "adapters.safetensors"
    if adapter_src.exists():
        dest = AILIANCE / "output" / "adapters" / model_name / domain
        dest.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(str(adapter_src), str(dest / "adapters.safetensors"))
        print(f"  Copied to {dest}")

    return True


def main():
    total = 0
    skipped = 0
    failed = 0

    for model_name, cfg in MODELS.items():
        print(f"\n{'='*60}")
        print(f"MODEL: {model_name} ({cfg['model'].split('/')[-1]})")
        print(f"{'='*60}")

        for domain in cfg["domains"]:
            if adapter_exists(model_name, domain):
                print(f"  SKIP {domain}: adapter already exists")
                skipped += 1
                continue

            ok = train_one(model_name, cfg, domain)
            if ok:
                total += 1
            else:
                failed += 1

    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {total} trained, {skipped} skipped, {failed} failed")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
