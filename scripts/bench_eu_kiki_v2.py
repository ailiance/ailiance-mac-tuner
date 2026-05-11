#!/usr/bin/env python3
"""AILIANCE v2 comprehensive benchmark.

1. Base vs LoRA (Qwen3.6 + Medium3.5)
2. Qwen36 vs Medium35 on shared domains
3. Full metrics: val_ppl, keyword_rate, response_len, degenerate_pct

Usage:
    python scripts/bench_eu_kiki_v2.py
    python scripts/bench_eu_kiki_v2.py --domains chat-fr cpp docker-devops
    python scripts/bench_eu_kiki_v2.py --qwen-only
    python scripts/bench_eu_kiki_v2.py --medium-only
"""

import argparse
import json
import time
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

import mlx.core as mx
from mlx_lm import load, generate

TUNNER = Path("/Users/clems/ailiance-mac-tuner")
DATA_DIR = Path("/Users/clems/ailiance/data/hf-traced")
CURRICULUM_DIR = TUNNER / "output/ailiance-v2-curriculum"
RESULTS_DIR = TUNNER / "results"

QWEN_MODEL = str(TUNNER / "models/Qwen3.6-35B-A3B-MLX-BF16")
MEDIUM_MODEL = str(TUNNER / "models/Mistral-Medium-3.5-128B-BF16")

DOMAIN_KEYWORDS = {
    "chat-fr": ["français", "expliqu", "fonction", "principe"],
    "cpp": ["std::", "#include", "template", "class "],
    "docker-devops": ["Dockerfile", "container", "docker", "deploy"],
    "embedded": ["GPIO", "UART", "I2C", "SPI", "interrupt"],
    "emc-dsp-power": ["EMI", "filter", "converter", "inductor"],
    "freecad": ["FreeCAD", "sketch", "Part", "parametric"],
    "html-css": ["CSS", "grid", "flexbox", "responsive"],
    "iot": ["MQTT", "sensor", "gateway", "protocol"],
    "kicad-dsl": ["symbol", "footprint", "pin", "schematic"],
    "kicad-pcb": ["PCB", "trace", "via", "copper", "layer"],
    "llm-ops": ["deploy", "inference", "model", "serve"],
    "llm-orch": ["agent", "chain", "prompt", "orchestrat"],
    "lua-upy": ["function", "require", "micropython", "lua"],
    "math-gsm8k": ["calculate", "total", "answer", "step"],
    "math-reasoning": ["prove", "theorem", "equation", "therefore"],
    "ml-training": ["epoch", "loss", "gradient", "batch", "train"],
    "multilingual-eu": ["traduction", "langue", "translation"],
    "music-audio": ["frequency", "audio", "signal", "waveform"],
    "platformio": ["platformio", "board", "upload", "serial"],
    "python": ["def ", "import ", "return", "class "],
    "rust": ["fn ", "let ", "impl ", "struct "],
    "rust-embedded": ["no_std", "cortex", "hal", "embedded"],
    "security-fenrir": ["vulnerability", "exploit", "secure", "audit"],
    "shell": ["#!/bin", "grep", "awk", "pipe"],
    "spice-sim": ["netlist", ".tran", "subckt", "ngspice"],
    "sql": ["SELECT", "FROM", "WHERE", "JOIN"],
    "stm32": ["STM32", "HAL", "DMA", "peripheral"],
    "traduction-tech": ["traduction", "technique", "terme"],
    "typescript": ["interface", "const ", "type ", "async"],
    "web-backend": ["API", "endpoint", "middleware", "route"],
    "web-frontend": ["React", "component", "render", "state"],
    "yaml-json": ["yaml", "json", "schema", "config"],
}


def load_valid_data(domain, max_samples=25):
    valid_file = DATA_DIR / domain / "valid.jsonl"
    if not valid_file.exists():
        return []
    samples = []
    with open(valid_file) as f:
        for line in f:
            if len(samples) >= max_samples:
                break
            try:
                obj = json.loads(line)
                msgs = obj.get("messages", [])
                if len(msgs) >= 2:
                    samples.append({
                        "prompt": msgs[0]["content"],
                        "reference": msgs[1]["content"],
                    })
            except json.JSONDecodeError:
                continue
    return samples


def compute_ppl(model, tokenizer, texts, max_samples=25):
    losses = []
    for text in texts[:max_samples]:
        tokens = mx.array(tokenizer.encode(text))
        if len(tokens) < 2:
            continue
        logits = model(tokens[None, :-1])
        targets = tokens[1:]
        loss = mx.mean(
            mx.losses.cross_entropy(logits.squeeze(0), targets)
        ).item()
        losses.append(loss)
        mx.eval(mx.zeros(1))  # sync
    if not losses:
        return 999.0
    import math
    avg_loss = sum(losses) / len(losses)
    return math.exp(min(avg_loss, 20))


def compute_keyword_rate(text, domain):
    keywords = DOMAIN_KEYWORDS.get(domain, [])
    if not keywords:
        return 0.0
    hits = sum(1 for kw in keywords if kw.lower() in text.lower())
    return hits / len(keywords)


def generate_response(model, tokenizer, prompt, max_tokens=256):
    try:
        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=max_tokens, verbose=False,
        )
        return response
    except Exception as e:
        return f"[ERROR] {e}"


def bench_domain(model, tokenizer, domain, samples, model_name=""):
    print(f"  {domain:<20} ", end="", flush=True)
    t0 = time.time()

    # 1. Perplexity
    references = [s["reference"] for s in samples]
    ppl = compute_ppl(model, tokenizer, references)

    # 2-4. Generation metrics (on first 5 prompts)
    keyword_rates = []
    resp_lens = []
    degenerate = 0
    gen_samples = samples[:5]

    for s in gen_samples:
        resp = generate_response(model, tokenizer, s["prompt"])
        kr = compute_keyword_rate(resp, domain)
        keyword_rates.append(kr)
        resp_lens.append(len(resp))
        if len(resp) < 10 or len(set(resp.split())) < 3:
            degenerate += 1

    elapsed = time.time() - t0
    result = {
        "domain": domain,
        "model": model_name,
        "val_ppl": round(ppl, 3),
        "avg_keyword_rate": round(sum(keyword_rates) / max(1, len(keyword_rates)), 3),
        "avg_resp_len": round(sum(resp_lens) / max(1, len(resp_lens)), 1),
        "degenerate_pct": round(degenerate / max(1, len(gen_samples)) * 100, 1),
        "elapsed_s": round(elapsed, 1),
    }

    print(f"ppl={ppl:>8.2f}  kw={result['avg_keyword_rate']:.2f}  "
          f"len={result['avg_resp_len']:>6.0f}  degen={result['degenerate_pct']:>4.0f}%  "
          f"({elapsed:.0f}s)")
    return result


def find_completed_domains(prefix):
    domains = []
    pattern = CURRICULUM_DIR / f"{prefix}-*"
    for d in sorted(CURRICULUM_DIR.glob(f"{prefix}-*")):
        if (d / "phase3_done").exists():
            domain = d.name.replace(f"{prefix}-", "")
            domains.append(domain)
    return domains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--qwen-only", action="store_true")
    ap.add_argument("--medium-only", action="store_true")
    ap.add_argument("--max-samples", type=int, default=25)
    args = ap.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    qwen_domains = find_completed_domains("qwen36")
    medium_domains = find_completed_domains("medium35")
    shared_domains = sorted(set(qwen_domains) & set(medium_domains))

    if args.domains:
        qwen_domains = [d for d in args.domains if d in qwen_domains]
        medium_domains = [d for d in args.domains if d in medium_domains]
        shared_domains = sorted(set(qwen_domains) & set(medium_domains))

    print("=" * 70)
    print(f"AILIANCE v2 BENCHMARK")
    print(f"  Qwen36 domains:  {len(qwen_domains)} {qwen_domains}")
    print(f"  Medium35 domains: {len(medium_domains)} {medium_domains}")
    print(f"  Shared (cross-model): {len(shared_domains)} {shared_domains}")
    print("=" * 70)

    all_results = {
        "qwen36_base": [],
        "qwen36_lora": [],
        "medium35_base": [],
        "medium35_lora": [],
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "qwen_model": QWEN_MODEL,
            "medium_model": MEDIUM_MODEL,
        },
    }

    # ===== QWEN36 =====
    if not args.medium_only and qwen_domains:
        print(f"\n{'='*70}")
        print(f"QWEN3.6-35B-A3B — BASE (no adapter)")
        print(f"{'='*70}")
        model, tok = load(QWEN_MODEL)

        for domain in qwen_domains:
            samples = load_valid_data(domain, args.max_samples)
            if not samples:
                print(f"  {domain:<20} SKIP (no valid data)")
                continue
            r = bench_domain(model, tok, domain, samples, "qwen36-base")
            all_results["qwen36_base"].append(r)

        del model, tok
        mx.metal.clear_cache()

        print(f"\n{'='*70}")
        print(f"QWEN3.6-35B-A3B — WITH LoRA ADAPTERS")
        print(f"{'='*70}")

        for domain in qwen_domains:
            adapter_path = str(CURRICULUM_DIR / f"qwen36-{domain}")
            samples = load_valid_data(domain, args.max_samples)
            if not samples:
                continue
            model, tok = load(QWEN_MODEL, adapter_path=adapter_path)
            r = bench_domain(model, tok, domain, samples, "qwen36-lora")
            all_results["qwen36_lora"].append(r)
            del model, tok
            mx.metal.clear_cache()

    # ===== MEDIUM35 =====
    if not args.qwen_only and medium_domains:
        print(f"\n{'='*70}")
        print(f"MEDIUM-3.5-128B — BASE (no adapter)")
        print(f"{'='*70}")
        model, tok = load(MEDIUM_MODEL)

        for domain in medium_domains:
            samples = load_valid_data(domain, args.max_samples)
            if not samples:
                print(f"  {domain:<20} SKIP (no valid data)")
                continue
            r = bench_domain(model, tok, domain, samples, "medium35-base")
            all_results["medium35_base"].append(r)

        del model, tok
        mx.metal.clear_cache()

        print(f"\n{'='*70}")
        print(f"MEDIUM-3.5-128B — WITH LoRA ADAPTERS")
        print(f"{'='*70}")

        for domain in medium_domains:
            adapter_path = str(CURRICULUM_DIR / f"medium35-{domain}")
            samples = load_valid_data(domain, args.max_samples)
            if not samples:
                continue
            model, tok = load(MEDIUM_MODEL, adapter_path=adapter_path)
            r = bench_domain(model, tok, domain, samples, "medium35-lora")
            all_results["medium35_lora"].append(r)
            del model, tok
            mx.metal.clear_cache()

    # ===== SUMMARY =====
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    for model_key in ["qwen36", "medium35"]:
        base_key = f"{model_key}_base"
        lora_key = f"{model_key}_lora"
        base = all_results[base_key]
        lora = all_results[lora_key]
        if not base or not lora:
            continue

        print(f"\n  {model_key.upper()}:")
        wins = {"lora": 0, "base": 0, "tie": 0}
        for b, a in zip(base, lora):
            delta = a["val_ppl"] - b["val_ppl"]
            w = "lora" if delta < -0.5 else "base" if delta > 0.5 else "tie"
            wins[w] += 1
            marker = "✓" if w == "lora" else "✗" if w == "base" else "="
            print(f"    {b['domain']:<20} base={b['val_ppl']:>8.2f}  "
                  f"lora={a['val_ppl']:>8.2f}  delta={delta:>+7.2f}  {marker}")

        avg_base = sum(r["val_ppl"] for r in base) / len(base)
        avg_lora = sum(r["val_ppl"] for r in lora) / len(lora)
        pct = (avg_base - avg_lora) / avg_base * 100
        print(f"    {'─'*55}")
        print(f"    Avg PPL: base={avg_base:.2f}  lora={avg_lora:.2f}  "
              f"improvement={pct:+.1f}%")
        print(f"    Wins: lora={wins['lora']}  base={wins['base']}  tie={wins['tie']}")

    # Cross-model comparison on shared domains
    if shared_domains and all_results["qwen36_lora"] and all_results["medium35_lora"]:
        print(f"\n  CROSS-MODEL (Qwen36 vs Medium35 LoRA on shared domains):")
        for domain in shared_domains:
            q = next((r for r in all_results["qwen36_lora"] if r["domain"] == domain), None)
            m = next((r for r in all_results["medium35_lora"] if r["domain"] == domain), None)
            if q and m:
                winner = "Qwen" if q["val_ppl"] < m["val_ppl"] else "Medium"
                print(f"    {domain:<20} Qwen={q['val_ppl']:>8.2f}  "
                      f"Medium={m['val_ppl']:>8.2f}  → {winner}")

    # Save results
    out_file = RESULTS_DIR / f"ailiance-v2-bench-{time.strftime('%Y%m%d-%H%M')}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
