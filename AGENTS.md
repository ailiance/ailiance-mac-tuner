# AGENTS.md

Guidance for AI coding agents (Claude Code, Aider, Cursor, etc.) working in this repo.

## Project

`KIKI-Mac_tunner` — LLM fine-tuning on Apple Silicon (Mac Studio M3 Ultra, 512 GB unified memory) via MLX. Distils Claude Opus reasoning into open-source models. Hosts the **Ailiance** foundation-distill tracks (Mistral Large, Qwen 35B/122B), Brainstacks 32-fleet, eu-kiki (3 EU/CH bases), SpikingKiki SNN, ANE hybrid, and the 32-domain meta-router. Repo `L-electron-Rare/KIKI-Mac_tunner`, branch `main`.

## Tech stack

- Language: Python 3 (Mac native)
- Runtime: pinned venv `~/mlx-stack/.venv` (snapshot 2026-05-10) — `pip install -r requirements.txt`
- Pinned core: `mlx==0.31.2`, `mlx-lm==0.31.3`, `mlx-metal==0.31.2`, `lm-eval==0.4.11`, `transformers==5.8.0`, `huggingface-hub==1.13.0`, `safetensors==0.7.0`, `numpy==2.4.4`
- Build/train: shell scripts as entry points (`setup.sh` → `download.sh` → `train.sh` → `export.sh`)
- Hardware: **Mac Studio M3 Ultra 512 GB** (bf16 full); some tracks (Brainstacks 4B) use RTX 4090
- Forks vendored: `lib/mlx_lm_fork/` (SSD offload); MLX 3× Metal limit fork installed in venv from `/tmp/mlx-fork`

## Commands

```bash
./setup.sh                       # bootstrap venv + tools
./download.sh                    # fetch models from HF
./train.sh                       # main training
./export.sh                      # convert + push HF
./distill-35b.sh                 # distill 35B teacher
bash train_eu_kiki_v2_curriculum.sh
python test_runtime_real.py
```

## Conventions

- Commits: subject ≤ 50 chars, body ≤ 72, no underscore in scope, no AI attribution, never `--no-verify`.
- Branches: `feat/<name>`, `fix/<name>`, `docs/<name>`, `eu-kiki/<name>`, `brainstacks/<name>`.
- Datasets: messages-format `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<thinking>...</thinking>\n\n..."}]}` — thinking blocks are part of the distill target.
- Apache-2.0 license repo-wide; published HF models are Apache-2.0 (datasets may carry upstream copyleft — keep CC-BY-SA / GPL when present).
- French in prose, English in code.

## File layout

- `configs/` — training/generation configs (incl. `eu-kiki-*.yaml`, `mlx-lm-micro-kiki-phase{1,2,3}.yaml`, `micro-kiki-router.yaml`, `micro_kiki/brainstacks.yaml`)
- `scripts/` — `train_*.py`, `distill*.sh`, `convert_*spiking*.py`, `quantize_spikingbrain.py`, `bench_*.py`, `eval_*.py`, `micro_kiki/`
- `src/` — runtime modules: `cognitive/`, `serving/`, `stacks/`
- `lib/mlx_lm_fork/` — SSD offload fork
- `research/ane-hybrid/` — DeltaNet → CoreML/ANE
- `data/` — datasets (large, often gitignored)
- `output/` — LoRA checkpoints
- `results/` — eval JSON (`v2-vs-v3.json`, `spikingkiki-*.json`)
- `docs/plans/`, `docs/specs/`, `docs/sota-training-2026.md`, `docs/micro-kiki-moe-research.md`
- `models/` — downloaded base models (large, gitignored)

## Domain-specific gotchas

- **Pinned MLX stack is intentional** — do not bump `mlx==0.31.2` / `mlx-lm==0.31.3` / `mlx-metal==0.31.2` without re-baselining all trained adapters. The `~/mlx-stack/.venv` snapshot is the production env.
- **Apple Silicon-only**. Non-Mac contributors cannot reproduce training runs locally; CI runs only inventory + smoke.
- **3× Metal buffer limit fork** is required for >70B bf16 training — installed from `/tmp/mlx-fork` (see L-electron-Rare/mlx `metal-Nx-buffer-limit` branches and memory file `reference_mlx_metal_buffer_fork.md`).
- **`thinking` blocks** in dataset assistant messages are part of the target; do not strip them in preprocessing — the distill target IS the reasoning trace.
- **eu-kiki ≠ kicad9plus**: per the Phase 6 scoreboard, `kicad9plus` causes **catastrophic forgetting** (-31 on P3). Only use it in KiCad-only contexts.
- **Mascarade LoRAs in `~/ailiance-models-tuning/outputs/`** (sister repo): all 10 are actually trained (audit 2026-05-18 corrected a stale "5/10 empty" note). Verify checkpoints before re-launching training to avoid waste.
- **HF org rename**: published artifacts moved `clemsail/eu-kiki-X` → `Ailiance-fr/X` (10/10 done). Cards under `Ailiance-fr` namespace are the canonical surface; old paths 307-redirect.
- **Router v6 (32-domain) is superseded by v7 (47-domain)** at the gateway — keep training data aligned with the v7 label set if producing router updates.
- **`/tmp/mlx-fork` is not in this repo** — clone L-electron-Rare/mlx first.

## When in doubt

- Read `CLAUDE.md` and `DATASETS_EMBEDDED_HARDWARE.md`.
- Recent commits: `git log --oneline -20`.
- Memory: `~/.claude/projects/-Users-electron/memory/project_ailiance_*`, `reference_mlx_metal_buffer_fork.md`, `reference_studio_post_reboot_2026_05_12.md`.
- Cluster context: `~/CLAUDE.md` (Mac Studio / Mac M1 / Tower / KXKM-AI infra block).
- Validate before pushing: `python test_runtime_real.py` and a small `bench_*.py` run.
