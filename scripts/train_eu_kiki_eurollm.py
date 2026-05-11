#!/usr/bin/env python3
"""Train EuroLLM 22B LoRA via mlx_lm_fork."""
import mlx.core as mx
mx.set_memory_limit(460 * 1024**3)
mx.set_cache_limit(32 * 1024**3)

import os, sys, yaml
from pathlib import Path

sys.path.insert(0, "/Users/clems/ailiance-mac-tuner/lib")

PROJECT = Path(__file__).parent.parent
OUTPUT = str(PROJECT / "output" / "ailiance" / "eurollm-chat-fr")
os.makedirs(OUTPUT, exist_ok=True)

config = {
    "model": str(PROJECT / "models" / "EuroLLM-22B-Instruct-2512"),
    "fine_tune_type": "lora",
    "lora_parameters": {"rank": 16, "alpha": 32, "dropout": 0.05, "scale": 2.0},
    "num_layers": -1,
    "learning_rate": 1e-5,
    "batch_size": 1,
    "grad_accumulation_steps": 4,
    "iters": 500,
    "max_seq_length": 2048,
    "grad_checkpoint": True,
    "save_every": 100,
    "steps_per_report": 5,
    "steps_per_eval": 100,
    "val_batches": 10,
    "train": True,
    "seed": 42,
}

config_path = Path(OUTPUT) / "train_config.yaml"
with open(config_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False)

from mlx_lm_fork.lora import main as lora_main
sys.argv = ["lora", "-c", str(config_path),
            "--data", str(PROJECT / "data" / "micro-kiki" / "chat-fr"),
            "--adapter-path", OUTPUT]
lora_main()
