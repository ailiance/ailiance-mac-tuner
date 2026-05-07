#!/usr/bin/env python3
"""Fuse MoE-LoRA adapters into base model using MLX (handles bf16 natively).

Strategy: Average all 4 expert LoRA pairs weighted by router bias,
add merged delta to base weight, save as standard HF safetensors.
"""
import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from safetensors.numpy import save_file as np_save_file
import numpy as np

BASE_MODEL = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/Qwen3.5-4B")
ADAPTER_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output/micro-kiki/stacks/python")
OUTPUT_DIR = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("output/micro-kiki/gguf/fused-model")

ALPHA = 32.0
RANK = 16
SCALING = ALPHA / RANK


def main():
    print(f"Base: {BASE_MODEL}")
    print(f"Adapter: {ADAPTER_PATH}")
    print(f"Output: {OUTPUT_DIR}")

    # Load adapter with MLX (handles bf16)
    adapter = mx.load(str(ADAPTER_PATH / "adapters.safetensors"))
    print(f"Adapter: {len(adapter)} tensors")

    # Copy non-safetensors files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in BASE_MODEL.iterdir():
        if f.suffix == ".safetensors":
            continue
        dst = OUTPUT_DIR / f.name
        if f.is_file() and not dst.exists():
            shutil.copy2(f, dst)
        elif f.is_dir() and not dst.exists():
            shutil.copytree(f, dst)

    proj_names = [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    ]

    fused_count = 0
    for shard_file in sorted(BASE_MODEL.glob("*.safetensors")):
        print(f"\n  {shard_file.name}...")
        base = mx.load(str(shard_file))

        modified = {}
        for key, weight in base.items():
            fused = weight

            # Try to match this key to a MoE-LoRA adapter
            for proj in proj_names:
                for layer_idx in range(36):  # Qwen3.5-4B has 36 layers
                    base_key = f"model.layers.{layer_idx}.{proj}.weight"
                    if key != base_key:
                        continue

                    adapter_prefix = f"language_model.model.layers.{layer_idx}.{proj}_moe_lora"

                    # Check if adapter exists for this projection
                    expert_a_key = f"{adapter_prefix}.experts.0.lora_a"
                    if expert_a_key not in adapter:
                        continue

                    # Get router weights for expert importance
                    router_bias_key = f"{adapter_prefix}.router_w2.bias"
                    if router_bias_key in adapter:
                        rb = adapter[router_bias_key].astype(mx.float32)
                        ew = mx.softmax(rb)
                    else:
                        ew = mx.ones(4) / 4.0

                    # Compute weighted delta
                    delta = mx.zeros_like(weight, dtype=mx.float32)
                    for i in range(4):
                        la = adapter[f"{adapter_prefix}.experts.{i}.lora_a"].astype(mx.float32)
                        lb = adapter[f"{adapter_prefix}.experts.{i}.lora_b"].astype(mx.float32)
                        expert_delta = (lb @ la) * SCALING
                        delta = delta + ew[i] * expert_delta

                    fused = (weight.astype(mx.float32) + delta).astype(weight.dtype)
                    fused_count += 1
                    break

            # Convert to numpy for saving (handles bf16→fp32 for compatibility)
            arr = np.array(fused.astype(mx.float16))
            modified[key] = arr

        # Save
        out_path = OUTPUT_DIR / shard_file.name
        np_save_file(modified, str(out_path))
        print(f"    Saved ({len(modified)} tensors, {fused_count} fused)")

    print(f"\nDone: {fused_count} projections fused")
    print(f"Output: {OUTPUT_DIR}")
    print(f"\nConvert to GGUF:")
    print(f"  python ~/llama.cpp/convert_hf_to_gguf.py {OUTPUT_DIR} --outfile output/micro-kiki/gguf/micro-kiki-v3-f16.gguf --outtype f16")
    print(f"  ~/llama.cpp/build/bin/llama-quantize output/micro-kiki/gguf/micro-kiki-v3-f16.gguf output/micro-kiki/gguf/micro-kiki-v3-Q4_K_M.gguf Q4_K_M")


if __name__ == "__main__":
    main()
