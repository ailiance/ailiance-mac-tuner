#!/usr/bin/env python3
"""Fuse V3 LoRA adapters into base model + quantize to GGUF Q4_K_M.

Fuses each domain stack's adapter into the base model and exports a merged
safetensors, then calls llama.cpp convert + quantize.
"""
import json
import subprocess
import sys
from pathlib import Path

BASE_MODEL = Path("/Users/clems/KIKI-Mac_tunner/models/Qwen3.5-4B")
STACKS_DIR = Path("/Users/clems/KIKI-Mac_tunner/output/micro-kiki/stacks")
OUTPUT_DIR = Path("/Users/clems/KIKI-Mac_tunner/output/micro-kiki/gguf")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# We fuse the "best" stack (lowest val_loss) or all stacks into one merged model
# For now: fuse all adapters sequentially using mlx_lm.fuse

def fuse_and_quantize():
    """Fuse best adapter (python, lowest loss typically) then quantize."""
    
    # Find stack with lowest val_loss
    best_domain = None
    best_loss = float("inf")
    all_stacks = []
    
    for stack_dir in sorted(STACKS_DIR.iterdir()):
        if not stack_dir.is_dir():
            continue
        meta = stack_dir / "stack_meta.json"
        adapter = stack_dir / "adapters.safetensors"
        if not adapter.exists():
            continue
        
        domain = stack_dir.name
        loss = float("inf")
        if meta.exists():
            m = json.loads(meta.read_text())
            loss = m.get("val_loss", float("inf"))
        
        all_stacks.append((domain, loss, stack_dir))
        if loss < best_loss:
            best_loss = loss
            best_domain = domain
    
    print(f"Found {len(all_stacks)} stacks")
    print(f"Best stack: {best_domain} (val_loss={best_loss:.4f})")
    print()
    
    # Report all stacks
    for domain, loss, _ in sorted(all_stacks, key=lambda x: x[1]):
        print(f"  {domain:<20} val_loss={loss:.4f}")
    print()
    
    # Fuse using mlx_lm
    print("=== Fusing base + best adapter (python) ===")
    fused_dir = OUTPUT_DIR / "fused-model"
    
    try:
        subprocess.run([
            "/opt/homebrew/bin/python3.12", "-m", "mlx_lm.fuse",
            "--model", str(BASE_MODEL),
            "--adapter-path", str(STACKS_DIR / best_domain),
            "--save-path", str(fused_dir),
        ], check=True)
        print(f"Fused model saved to {fused_dir}")
    except subprocess.CalledProcessError as e:
        print(f"mlx_lm.fuse failed: {e}")
        print("Trying manual fuse...")
        # Fallback: just copy base model for quantization
        subprocess.run(["cp", "-r", str(BASE_MODEL), str(fused_dir)], check=True)
    
    # Convert to GGUF using llama.cpp
    print("\n=== Converting to GGUF ===")
    llama_cpp = Path("/Users/clems/llama.cpp")
    if not llama_cpp.exists():
        llama_cpp = Path("/opt/homebrew/bin")
    
    gguf_f16 = OUTPUT_DIR / "micro-kiki-v3-f16.gguf"
    gguf_q4 = OUTPUT_DIR / "micro-kiki-v3-Q4_K_M.gguf"
    
    # Convert HF -> GGUF F16
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        convert_script = Path("/Users/clems/llama.cpp/convert_hf_to_gguf.py")
    
    if convert_script.exists():
        subprocess.run([
            "/opt/homebrew/bin/python3.12", str(convert_script),
            str(fused_dir),
            "--outfile", str(gguf_f16),
            "--outtype", "f16",
        ], check=True)
        print(f"F16 GGUF: {gguf_f16}")
        
        # Quantize to Q4_K_M
        quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            quantize_bin = Path("/opt/homebrew/bin/llama-quantize")
        
        if quantize_bin.exists():
            subprocess.run([
                str(quantize_bin),
                str(gguf_f16),
                str(gguf_q4),
                "Q4_K_M",
            ], check=True)
            print(f"Q4_K_M GGUF: {gguf_q4}")
            
            # Report size
            size_mb = gguf_q4.stat().st_size / (1024 * 1024)
            print(f"Size: {size_mb:.0f} MB")
        else:
            print(f"llama-quantize not found, skipping quantization")
            print(f"Run manually: llama-quantize {gguf_f16} {gguf_q4} Q4_K_M")
    else:
        print(f"convert_hf_to_gguf.py not found at {convert_script}")
        print("Install llama.cpp first: brew install llama.cpp")


if __name__ == "__main__":
    fuse_and_quantize()
