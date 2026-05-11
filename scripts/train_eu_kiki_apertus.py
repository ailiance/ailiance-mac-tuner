#!/usr/bin/env python3
"""Train Apertus-70B LoRA via mlx_lm_fork (Metal buffer patches).

Usage:
    cd ~/ailiance-mac-tuner
    .venv/bin/python scripts/train_eu_kiki_apertus.py

Requires: sudo sysctl -w iogpu.wired_limit_mb=458752
"""
import mlx.core as mx
mx.set_memory_limit(480 * 1024**3)
mx.set_cache_limit(32 * 1024**3)

import os
import sys
import yaml
from pathlib import Path

# Use the fork with Metal buffer patches
sys.path.insert(0, "/Users/clems/ailiance-mac-tuner/lib")

PROJECT = Path(__file__).parent.parent
MODEL = str(PROJECT / "models" / "Apertus-70B-Instruct-2509")
DATA = str(PROJECT / "data" / "micro-kiki" / "electronics")
OUTPUT = str(PROJECT / "output" / "ailiance" / "apertus-electronics")

os.makedirs(OUTPUT, exist_ok=True)

config = {
    "model": MODEL,
    "fine_tune_type": "lora",
    "lora_parameters": {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.05,
        "scale": 2.0,
    },
    "num_layers": -1,
    "learning_rate": 1e-5,
    "batch_size": 1,
    "grad_accumulation_steps": 8,
    "iters": 500,
    "max_seq_length": 1024,
    "grad_checkpoint": True,
    "save_every": 100,
    "steps_per_report": 5,
    "steps_per_eval": 100,
    "val_batches": 5,
    "train": True,
    "seed": 42,
}

config_path = Path(OUTPUT) / "train_config.yaml"
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"Config: {config_path}")
print(f"Model: {MODEL}")
print(f"Data: {DATA}")
print(f"Output: {OUTPUT}")

from mlx_lm_fork.lora import main as lora_main

sys.argv = [
    "lora", "-c", str(config_path),
    "--data", DATA,
    "--adapter-path", OUTPUT,
]
lora_main()
