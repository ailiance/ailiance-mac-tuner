#!/usr/bin/env python3
"""
ailiance-mac-tuner — MLX LoRA fine-tuning for Apple Silicon.

Designed for Mac Studio M3 Ultra / M4 Pro 512 Go.
Supports bf16 full precision LoRA on models up to ~250B parameters.

Usage:
    python scripts/train_mlx.py --config configs/mistral-large.yaml
    python scripts/train_mlx.py --config configs/mistral-large.yaml --resume
"""

import argparse
import os
import sys
import time
import signal
import yaml
import json
import math
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.tuner.trainer import TrainingArgs, TrainingCallback, CacheDataset, train as mlx_train
from mlx_lm.tuner.datasets import load_local_dataset
from datasets import load_dataset as hf_load_dataset


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def format_dataset(config: dict, script_dir: Path) -> Path:
    """Format the HF dataset into JSONL for mlx-lm."""
    dataset_id = config["dataset_id"]
    data_dir = script_dir / "data" / Path(dataset_id).name
    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"

    if train_file.exists() and valid_file.exists():
        n_train = sum(1 for _ in open(train_file))
        print(f"  Dataset already formatted: {n_train} train examples")
        return data_dir

    print(f"  Downloading and formatting {dataset_id}...")
    ds = hf_load_dataset(dataset_id, split="train")

    # Format as chat conversations with thinking/reasoning
    formatted = []
    for example in ds:
        conversation = {
            "messages": [
                {"role": "user", "content": example["problem"]},
                {
                    "role": "assistant",
                    "content": (
                        f"<thinking>\n{example['thinking']}\n</thinking>\n\n"
                        f"{example['solution']}"
                    ),
                },
            ]
        }
        formatted.append(json.dumps(conversation, ensure_ascii=False))

    # 95/5 train/valid split
    split_idx = int(len(formatted) * 0.95)
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(train_file, "w") as f:
        f.write("\n".join(formatted[:split_idx]))
    with open(valid_file, "w") as f:
        f.write("\n".join(formatted[split_idx:]))

    print(f"  Train: {split_idx} | Valid: {len(formatted) - split_idx}")
    return data_dir


def apply_lora(model, config: dict):
    """Apply LoRA adapters to all linear layers."""
    rank = config.get("lora_rank", 96)
    alpha = config.get("lora_alpha", 192)
    dropout = config.get("lora_dropout", 0.05)
    scale = alpha / rank

    lora_layers = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.shape[0] > 1:
            # Replace with LoRA
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model
            for part in parent_name.split("."):
                if part:
                    if isinstance(parent, list):
                        parent = parent[int(part)]
                    else:
                        parent = getattr(parent, part)

            lora_linear = LoRALinear.from_base(
                module, r=rank, scale=scale, dropout=dropout
            )
            if isinstance(parent, list):
                parent[int(child_name)] = lora_linear
            else:
                setattr(parent, child_name, lora_linear)
            lora_layers += 1

    from mlx.utils import tree_flatten
    all_params = tree_flatten(model.parameters())
    total_params = sum(p.size for _, p in all_params)
    trainable = sum(
        p.size for name, p in all_params if "lora" in name.lower()
    )
    print(f"  LoRA applied: {lora_layers} layers, rank={rank}, alpha={alpha}")
    print(
        f"  Parameters: {total_params:,} total, {trainable:,} trainable "
        f"({trainable/total_params*100:.2f}%)"
    )
    return model


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find the latest checkpoint directory."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    checkpoints = sorted(
        [d for d in output_path.iterdir() if d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


class SafetyCallback(TrainingCallback):
    """Detects NaN/Inf in loss and implements early stopping."""

    def __init__(self, nan_patience: int = 3, early_stopping_patience: int = 3):
        self.nan_count = 0
        self.nan_patience = nan_patience
        self.best_val_loss = float("inf")
        self.es_counter = 0
        self.es_patience = early_stopping_patience
        self.best_step = 0

    def on_train_loss_report(self, train_info: dict):
        loss = train_info.get("train_loss", 0.0)
        step = train_info.get("iteration", 0)
        if math.isnan(loss) or math.isinf(loss):
            self.nan_count += 1
            print(f"\n  WARNING: NaN/Inf in train_loss at step {step} "
                  f"({self.nan_count}/{self.nan_patience})")
            if self.nan_count >= self.nan_patience:
                print("  ABORTING: Too many NaN/Inf losses. "
                      "Try reducing learning_rate or check bf16 stability.")
                raise KeyboardInterrupt("NaN detected")
        else:
            self.nan_count = 0

    def on_val_loss_report(self, val_info: dict):
        val_loss = val_info.get("val_loss", 0.0)
        step = val_info.get("iteration", 0)
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.es_counter = 0
            self.best_step = step
        else:
            self.es_counter += 1
            print(f"  EarlyStopping: val_loss {val_loss:.4f} did not improve "
                  f"({self.es_counter}/{self.es_patience}), "
                  f"best={self.best_val_loss:.4f} at step {self.best_step}")
            if self.es_counter >= self.es_patience:
                print(f"  STOPPING: val_loss has not improved for "
                      f"{self.es_patience} evals. "
                      f"Best: {self.best_val_loss:.4f} at step {self.best_step}")
                raise KeyboardInterrupt("Early stopping triggered")


def main():
    parser = argparse.ArgumentParser(description="ailiance-mac-tuner — MLX LoRA training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mistral-large.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent.parent
    config = load_config(script_dir / args.config)

    print("=" * 60)
    print("ailiance-mac-tuner — MLX LoRA Fine-tuning")
    print(f"Config: {args.config}")
    print(f"Model: {config['model_id']}")
    print(f"Precision: {config.get('precision', 'bf16')}")
    print(f"LoRA: rank={config['lora_rank']}, alpha={config['lora_alpha']}")
    print("=" * 60)

    # Check memory and set limits
    try:
        import subprocess
        mem_bytes = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()
        )
        mem_gb = mem_bytes / 1024**3
        print(f"\nSystem memory: {mem_gb:.0f} GB")
        if mem_gb < 256:
            print(
                "WARNING: <256 GB RAM. Consider using 4-bit quantized training."
            )
        # Cap Metal memory usage (leave headroom for OS)
        mem_limit_gb = config.get("memory_limit_gb", 460)
        mx.set_memory_limit(mem_limit_gb * 1024**3)
        mx.set_cache_limit(32 * 1024**3)
        print(f"Metal memory limit: {mem_limit_gb} GB, cache limit: 32 GB")
    except Exception:
        pass

    # Format dataset
    print("\n[1/4] Preparing dataset...")
    data_dir = format_dataset(config, script_dir)

    # Load model
    model_dir = script_dir / "models" / Path(config["model_id"]).name
    if not model_dir.exists():
        print(f"\n[2/4] Downloading model {config['model_id']}...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            config["model_id"],
            local_dir=str(model_dir),
            ignore_patterns=["*.bin", "*.pt"],
        )
    else:
        print(f"\n[2/4] Loading model from {model_dir}...")

    model, tokenizer = load(str(model_dir))

    # Apply LoRA
    print("\n[3/4] Applying LoRA adapters...")
    model = apply_lora(model, config)

    # Resume
    resume_adapter = None
    if args.resume:
        checkpoint = find_latest_checkpoint(
            str(script_dir / config["output_dir"])
        )
        if checkpoint:
            print(f"  Resuming from: {checkpoint}")
            resume_adapter = checkpoint
        else:
            print("  No checkpoint found, starting fresh")

    # Load datasets
    print("\n[4/4] Training...")
    output_dir = script_dir / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # load_local_dataset expects a config-like object
    class DataConfig:
        def __init__(self):
            self.mask_prompt = False
    data_config = DataConfig()
    train_raw, val_raw, _ = load_local_dataset(data_dir, tokenizer, data_config)
    train_dataset = CacheDataset(train_raw)
    val_dataset = CacheDataset(val_raw)

    # Estimate total iterations from dataset size
    n_train = len(train_dataset)
    batch_size = config.get("batch_size", 1)
    grad_accum = config.get("gradient_accumulation_steps", 4)
    num_epochs = config.get("num_epochs", 2)
    total_iters = (n_train * num_epochs) // (batch_size * grad_accum)
    print(f"  Dataset: {n_train} train examples")
    print(f"  Total iterations: {total_iters} ({num_epochs} epochs)")
    print(f"  Effective batch size: {batch_size * grad_accum}")

    train_args = TrainingArgs(
        batch_size=batch_size,
        iters=total_iters,
        val_batches=config.get("val_batches", 5),
        steps_per_report=5,
        steps_per_eval=config.get("save_every", 50),
        steps_per_save=config.get("save_every", 50),
        max_seq_length=config.get("max_seq_length", 4096),
        adapter_file=str(output_dir / "adapters.safetensors"),
        grad_checkpoint=True,
        grad_accumulation_steps=grad_accum,
    )

    # Build optimizer with cosine schedule
    lr = config.get("learning_rate", 1.5e-5)
    warmup_steps = int(total_iters * config.get("warmup_ratio", 0.05))
    schedule = optim.join_schedules(
        [optim.linear_schedule(1e-7, lr, warmup_steps),
         optim.cosine_decay(lr, total_iters - warmup_steps)],
        [warmup_steps],
    )
    optimizer = optim.AdamW(
        learning_rate=schedule,
        weight_decay=config.get("weight_decay", 0.01),
    )

    print(f"  Learning rate: {lr} (warmup {warmup_steps} steps, cosine decay)")

    # Resume adapter weights if applicable
    if resume_adapter:
        model.load_weights(str(Path(resume_adapter) / "adapters.safetensors"), strict=False)

    # Build safety callback (NaN detection + early stopping)
    early_stopping_patience = config.get("early_stopping_patience", 3)
    callback = SafetyCallback(
        nan_patience=3,
        early_stopping_patience=early_stopping_patience,
    )

    # Use mlx-lm's built-in trainer
    mlx_train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        args=train_args,
        training_callback=callback,
    )

    # Save final adapter
    final_dir = output_dir / "final-lora"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(final_dir / "adapters.safetensors"))
    print(f"\n=== Training complete! ===")
    print(f"Final LoRA adapter: {final_dir}")
    print(f"Next: ./export.sh")


if __name__ == "__main__":
    main()
