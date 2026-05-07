#!/usr/bin/env python3
"""Full benchmark: Base 35B vs LoRA 35B vs LoRA 4B on 35 domains."""
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

PROMPTS = {
    "chat-fr": "Explique le fonctionnement d un variateur de frequence pour moteur asynchrone triphase.",
    "reasoning": "Three resistors 100 200 300 ohms in parallel. Calculate total resistance step by step.",
    "python": "Write a Python decorator that retries a function with exponential backoff and jitter.",
    "typescript": "Write a TypeScript generic type-safe event emitter with on off and emit methods.",
    "cpp": "Write a C++ lock-free SPSC ring buffer using std atomic.",
    "rust": "Write a Rust function that parses CSV into Vec of structs using serde.",
    "html-css": "Write a responsive CSS Grid dashboard layout with sidebar header main and footer.",
    "shell": "Write a bash script that audits all git repos under a directory for uncommitted changes.",
    "sql": "Write a PostgreSQL CTE with window functions for running 7-day average revenue.",
    "yaml-json": "Write a GitHub Actions CI workflow for Python with matrix testing and caching.",
    "docker": "Write a multi-stage Dockerfile for a Rust web service with cargo-chef layer caching.",
    "kicad-dsl": "Write a KiCad Python plugin that generates a spiral inductor footprint.",
    "spice": "Write a complete ngspice netlist for a class-D audio amplifier with THD measurement.",
    "lua-upy": "Write a MicroPython BME280 driver over I2C on ESP32 with altitude calculation.",
    "embedded": "Write ESP-IDF code for FreeRTOS task reading INA226 via I2C with moving average.",
    "stm32": "Write STM32 HAL code for DMA circular ADC multi-channel with interrupts.",
    "iot": "Design an MQTT topic hierarchy for 1000 battery sensors with OTA updates.",
    "freecad": "Write a FreeCAD Python macro for a parametric enclosure with snap-fit lid.",
    "platformio": "Write platformio.ini for multi-target ESP32 plus STM32 with shared libs.",
    "power": "Design a synchronous buck converter 12V to 3.3V at 5A. Calculate inductor and caps.",
    "emc": "Explain PCB ground plane strategy for mixed-signal analog digital RF to minimize EMI.",
    "dsp": "Implement a 4th order IIR Butterworth bandpass 300-3400Hz in C for ARM Cortex-M4.",
    "spice-sim": "Write ngspice control script for parametric sweep of R C with 3dB frequency extraction.",
    "electronics": "Compare bootstrap vs charge pump vs isolated gate drivers for 3-phase inverter.",
    "kicad-pcb": "Explain DFM review checklist for 4-layer PCB: trace width via rules copper pour.",
    "web-frontend": "Build a React 19 data table with server-side sort filter and pagination.",
    "web-backend": "Design a FastAPI webhook system with PostgreSQL queue retry and HMAC verification.",
    "music-audio": "Implement a polyphonic wavetable synthesizer in Python with ADSR and LFO.",
    "devops": "Write Terraform for AWS ECS Fargate with ALB auto-scaling and blue-green deploy.",
    "llm-orch": "Build a RAG pipeline with hybrid BM25 plus vector retrieval and cross-encoder reranking.",
    "math": "Derive transfer function of Sallen-Key low-pass filter and find Q factor.",
    "security": "Implement PKCE OAuth2 flow with JWT rotation and CSRF protection for SPA plus API.",
    "components": "Compare STM32F103 ESP32-S3 RP2040 for battery IoT: power peripherals price ecosystem.",
    "llm-ops": "Set up vLLM with tensor parallelism Prometheus metrics and automatic model switching.",
    "ml-training": "Write LoRA fine-tuning script with cosine LR warmup grad checkpoint and W&B logging.",
}


def compute_ppl(model, tokenizer, prompt):
    tokens = tokenizer.encode(prompt)
    if len(tokens) < 2:
        return 999.0
    ids = mx.array([tokens[:-1]])
    labels = mx.array([tokens[1:]])
    logits = model(ids)
    loss = nn.losses.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
    )
    return float(mx.exp(mx.mean(loss)).item())


def main():
    results = []
    out_dir = Path("output/micro-kiki/eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base 35B
    print("=== BASE 35B (no adapter) ===", flush=True)
    model, tok = load("models/Qwen3.6-35B-A3B")
    for domain, prompt in PROMPTS.items():
        ppl = compute_ppl(model, tok, prompt)
        results.append({"domain": domain, "model": "base-35b", "ppl": round(ppl, 2)})
        print(f"  {domain:<18} ppl={ppl:.2f}", flush=True)
    del model
    mx.metal.reset_peak_memory()

    # 2. 35B + LoRA
    print("\n=== 35B + LoRA ===", flush=True)
    for domain, prompt in PROMPTS.items():
        adapter = f"output/micro-kiki/lora-qwen36-35b/{domain}"
        if not Path(adapter).exists():
            results.append({"domain": domain, "model": "lora-35b", "ppl": 999})
            continue
        model, tok = load("models/Qwen3.6-35B-A3B", adapter_path=adapter)
        ppl = compute_ppl(model, tok, prompt)
        results.append({"domain": domain, "model": "lora-35b", "ppl": round(ppl, 2)})
        print(f"  {domain:<18} ppl={ppl:.2f}", flush=True)
        del model

    # 3. 4B + LoRA
    print("\n=== 4B + LoRA ===", flush=True)
    for domain, prompt in PROMPTS.items():
        adapter = f"output/micro-kiki/lora-standard/{domain}"
        if not Path(adapter).exists():
            results.append({"domain": domain, "model": "lora-4b", "ppl": 999})
            continue
        model, tok = load("models/Qwen3.5-4B", adapter_path=adapter)
        ppl = compute_ppl(model, tok, prompt)
        results.append({"domain": domain, "model": "lora-4b", "ppl": round(ppl, 2)})
        print(f"  {domain:<18} ppl={ppl:.2f}", flush=True)
        del model

    # Save
    with open(out_dir / "bench-35b-vs-4b.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    hdr = f"{'Domain':<18} {'Base 35B':>10} {'LoRA 35B':>10} {'LoRA 4B':>10} {'Winner':>10}"
    print(hdr)
    print("-" * 70)

    wins = {"base-35b": 0, "lora-35b": 0, "lora-4b": 0}
    for domain in PROMPTS:
        base = next((r["ppl"] for r in results if r["domain"] == domain and r["model"] == "base-35b"), 999)
        l35 = next((r["ppl"] for r in results if r["domain"] == domain and r["model"] == "lora-35b"), 999)
        l4 = next((r["ppl"] for r in results if r["domain"] == domain and r["model"] == "lora-4b"), 999)
        best = min(base, l35, l4)
        if best == l35:
            w = "lora-35b"
        elif best == l4:
            w = "lora-4b"
        else:
            w = "base-35b"
        wins[w] += 1
        print(f"{domain:<18} {base:>10.2f} {l35:>10.2f} {l4:>10.2f} {w:>10}")

    print("-" * 70)
    for m, c in wins.items():
        print(f"  {m}: {c} wins")
    print(f"\nSaved to {out_dir / 'bench-35b-vs-4b.json'}")


if __name__ == "__main__":
    main()
