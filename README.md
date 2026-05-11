# ailiance-mac-tuner

Fine-tune large language models on Apple Silicon using the full unified memory — no quantization needed.

## What This Does

Distills Claude Opus reasoning into open-source LLMs (Mistral Large 123B, Qwen3.5-122B-A10B, Devstral 2 123B) using MLX on a single Mac Studio. BF16 full-precision training with LoRA, enabled by 512 GB unified memory.

Includes:
- **AILIANCE v2** — 3-phase curriculum training (seq 512→1280→4096) on Qwen3.6-35B-A3B (31 domains) + Mistral Medium 3.5 128B (33 domains)
- **Brainstacks** — 32-expert MoE-LoRA fleet on Qwen3.5 with null-space projection (zero-forgetting continual learning)
- **ailiance v1** — EU-sovereign training stack (Apertus 70B / Devstral 2 24B / EuroLLM 22B) with EU AI Act-compliant dataset provenance
- **SpikingKiki** — LoRA → spiking neural network conversion track (LAS rate-coded)
- **ANE hybrid pipeline** — DeltaNet on Apple Neural Engine for hybrid inference
- **Meta-router** — 32-domain attention-pooling router for stack dispatch
- **VLM support** — mlx-vlm for vision-language model training and inference on Apple Silicon

## Machine

- **Mac Studio M3 Ultra** — 512 GB unified memory
- macOS 15+ (Sequoia)
- MLX with custom 3× Metal buffer limit (499K → 1.5M buffers) for 122B BF16 training

## Quick Start

```bash
./setup.sh          # install dependencies (mlx, mlx-lm, mlx-tune)
./download.sh       # download model + dataset
./train.sh          # launch training (Ctrl+C saves checkpoint)
./train.sh --resume # resume from last checkpoint
./export.sh         # merge LoRA + convert to GGUF
```

For the 122B MoE model:
```bash
sudo sysctl -w iogpu.wired_limit_mb=458752   # cap Metal at 448 GiB
./scripts/train_122b_macport.sh               # launch 122B training
```

## Training Results

> **Reality-check 2026-05-04** — see [`docs/CLAUDE.md`](docs/CLAUDE.md) Roadmap section for priorities. Several published statuses below were **CRASHED**, not Done.

### Foundation distillation (Opus → open-source)

| Model | Method | Val Loss | Status | Notes |
|-------|--------|----------|--------|-------|
| Mistral Large 123B | LoRA bf16 | **0.479** | **DONE** (iter 1100, dormant since Apr 13) | `output/mistral-large-opus/adapters.safetensors` 3.36 GB — never merged or published |
| Qwen3.5-122B-A10B-Opus-v3 | mlx-tune LoRA bf16 | crashed @ iter 1 | **CRASHED OOM** | `logs/curriculum.log` Insufficient Memory at iter 1 — needs seq_length reduction or MLX recompile |
| Qwen3.5-35B-A3B-Opus-v3 | LoRA bf16 | — | **NOT STARTED** | dir `output/qwen35-35b-opus-final/` contains only config |
| Qwen3.5-35B-A3B-Opus-VLM | mlxvlm distill | crashed | **CRASHED** | `distill-mlxvlm-resume.log` 0 bytes |
| Mistral Small Opus distill | QLoRA Unsloth | — | **PLAN ONLY** (target: kxkm-ai RTX 4090) | `docs/plans/2026-04-15-mistral-small-opus-distill.md` |
| Devstral v4 Small 2 24B Opus | SFT+SimPO Unsloth | — | **PLAN ONLY** (target: kxkm-ai) | `docs/plans/2026-04-15-devstral-v4-small-2-opus-distill.md` |

Peak memory for 122B training: 383 GB (failed to allocate beyond).

### Brainstacks fleet — 3 iterations

| Version | Path | Stacks | Status |
|---------|------|--------|--------|
| v3 r=16 | `output/micro-kiki/stacks-v3-r16/` | 32 × 459 MB | **DEAD** — `lora_B = 0` partout, val_loss identique = 1.28. Stacks effondrés (cf. `stack_meta.json`). To archive. |
| v4 dynrank | `output/micro-kiki/stacks-v4-dynrank/` | 31 × 297-1431 MB | **PARTIAL** — 31/32 (manque `components` + 6 v2) |
| v4 sota | `output/micro-kiki/lora-qwen36-35b-v4-sota/` | 39 dirs (33 base + 6 v2) | **TRAINED** Apr 19-26, ~130 GB total — **never re-evaluated since training completed** |

→ **Canonical version not yet selected.** Last `bench-complete.json` is Apr 19, antérieur à la fin de v4-sota (Apr 26).

### V2 vs V3 head-to-head (35B, 10 niches)

`results/v2-vs-v3.json` (2026-04-17) — `output/micro-kiki/eval/fused_eval_results.json` shows V2 et V3 perplexités **strictement identiques bit-à-bit** (chat-fr 31.17/31.17, python 5.75/5.75…).

> ⚠️ **Bug d'éval probable** : `eval_v2_v3.py` rend les mêmes scores → soit l'adapter n'est pas chargé, soit les deux versions sont littéralement le même artefact. À débugger.

### SpikingKiki conversion

`results/spikingkiki-35b-convert.json` (2026-04-22) — Qwen3.6-35B-A3B → SpikingKiki-35B-V4: 31070 layers converted, 128 timesteps, 11086 s, status "ok". 58 GB sur disque (`models/SpikingKiki-35B-A3B-V4/`).

> ⚠️ **Suite non lancée** : Q4 quantization, smoke test (`smoke_spikingbrain.py`), energy bench ANN vs SNN — aucun log post-conversion.

### AILIANCE v2 — curriculum training (active)

3-phase curriculum: seq 512→1280→4096, LR 8e-6→5e-6→3e-6. Resume-safe (skips completed phases).

| Model | Domains | LoRA config | Phase 3 seq | Status (2026-05-07) |
|-------|---------|-------------|-------------|---------------------|
| Qwen3.6-35B-A3B BF16 | 31 | r32/α32, 16 layers | 4096 | **2/31 complete**, 4 partial, 25 not started (~30h remaining) |
| Mistral Medium 3.5 128B BF16 | 33 | r16/α32, all layers | 2048 | **0/33** — starts after Qwen36 completes |

Scripts: `train_eu_kiki_v2_curriculum.sh` (3-phase), `train_eu_kiki_v2_batch.sh` (flat), `train_eu_kiki_v2_retry_failed.sh` (Metal-crash recovery, batch=1).
Configs: `configs/ailiance-v2-qwen36-*.yaml` (per-domain probes).
Output: `output/ailiance-v2-curriculum/{qwen36,medium35}-<domain>/`.
Data: `~/ailiance/data/hf-traced/` (36 domains).

#### Adaptive curriculum (iters / dropout vs dataset size)

Detected at start of each domain by counting `train.jsonl` lines. Anti-overfit on small domains, full schedule on large ones.

| Dataset size | Phase 1 iters | Phase 2 iters | Phase 3 iters | Dropout |
|--------------|---------------|---------------|---------------|---------|
| `< 100`      | 100           | 150           | 80            | 0.08    |
| `< 500`      | 200           | 300           | 150           | 0.05    |
| `< 1000`     | 300           | 500           | 250           | 0.03    |
| `≥ 1000`     | 500           | 800           | 500           | 0.01 (standard) |

Implemented in `train_eu_kiki_v2_curriculum.sh`. Domains without `train.jsonl` are silently skipped (P2 hardening: add explicit warning).

### ailiance v1 training (sister project)

| Model | Adapters trained | Total target | Status |
|-------|------------------|--------------|--------|
| Apertus 70B | 6 (electronics-hw, embedded, math, math-gsm8k, math-reasoning, spice-sim) | 8 | **PARTIAL** — manque `emc-dsp-power`, `security-fenrir` |
| Devstral 2 24B | 22 | 22 | **DONE** |
| EuroLLM 22B | 3 (chat-fr, multilingual-eu, traduction-tech) | 4 | **PARTIAL** |
| Router 32-domain | trained | — | **DONE** (`~/ailiance/output/router/router.safetensors`) |
| Eval framework | code prêt (52 ko, EU AI Act Art. 53(1)(d)) | — | **JAMAIS LANCÉ** — `output/eval/raw/` vide |

### HuggingFace publications

Post-2026-05-11 ailiance carve-out : adapters et datasets de production publiés
sous l'org [`Ailiance-fr`](https://huggingface.co/Ailiance-fr).

**`Ailiance-fr/` (org)** — 15 modèles + 13 datasets
- 10 × adapters Apache-2.0 (devstral×4, apertus×3, eurollm, routers minilm×2)
- 5 × `qwen3-4b-mascarade-{kicad,spice,stm32,emc,embedded}-lora`
  (validés bench Phase 6, voir [`ailiance/ailiance-bench`](https://github.com/ailiance/ailiance-bench))
- 13 datasets `mascarade-*-dataset` + `kicad9plus-{permissive,copyleft}` + `kill-life-embedded-qa`

**Legacy `clemsail/` (perso)** — rename-ghosts post-migration
- 16 modèles historiques redirigent vers `Ailiance-fr/` (HF rename redirects)
- 5 × à 0 dl restent legacy (`micro-kiki-{v35b,router-v4,v4-sota}`, `spikingkiki-{35b-a3b-v4,v4-adapters}`) — model cards à enrichir ou archiver

**Legacy `electron-rare/` (org)** — archive post-carve-out
- 8 datasets résiduels avec disclosure warnings (voir mémoire `project_ailiance_org_2026_05_11.md`)
- Tous les datasets actifs sont publiés sous `Ailiance-fr/`

**Pas encore publiés**
- `output/mistral-large-opus/adapters.safetensors` (3.36 GB, terminé iter 1100, dormant 21+ j)
- Stack AILIANCE v1 (Apertus 70B + Devstral 24B + EuroLLM 22B) tourne en prod via
  [`ailiance/ailiance`](https://github.com/ailiance/ailiance), modèles base upstream non re-publiés

Le script `scripts/release_hf.py` reste en mode dry-run, jamais lancé avec
`--execute` (les publications ont été faites manuellement ou via les scripts
de migration 2026-05-11 dans `ailiance/ailiance`).

## Inference Benchmarks

| Model | Engine | Throughput |
|-------|--------|------------|
| Qwen3.5-35B-A3B | mlx-vlm native | 45–89 tok/s |
| DeltaNet 40-layer (ANE) | CoreML | 14.4 tok/s (474 tok/s/layer) |
| MLX pure (full model) | MLX | 14.2 tok/s |
| ANE+CPU hybrid | CoreML+MLX | 9.9 tok/s |

## Datasets

| Dataset | Examples |
|---------|----------|
| combined-opus-14k (deduplicated) | 9,813 |
| final-opus-v3-1 | 11,880 train + 626 valid |
| Distilled (123B + 35B + vlm) | ~2,237 |
| Brainstacks raw → deduplicated | 1.57M → 63K |
| Devstral-Sonnet (R1 + SWE) | ~18K |

See [`DATASETS_EMBEDDED_HARDWARE.md`](DATASETS_EMBEDDED_HARDWARE.md) for embedded/hardware dataset research.

## World Firsts

### 1. DeltaNet → CoreML/ANE Conversion
First-ever conversion of Gated DeltaNet (linear attention with recurrent state) to CoreML for Apple Neural Engine. No prior work existed — ANEMLL only supports standard transformer attention.

- Chunkwise parallel form expressed as CoreML MIL ops (matmul, cumsum, exp)
- `ct.StateType` for recurrent state persistence between decode steps
- **474 tok/s per layer**, 14.4 tok/s for full 40-layer stack on ANE
- Real Qwen3.5 weights loaded and verified

### 2. 122B MoE BF16 Training on Single Apple Silicon Machine
First documented fine-tuning of a 122B MoE model in BF16 on a single Mac. Previous record: dense 20B on 512 GB.

- Qwen3.5-122B-A10B (10B active params) at 383 GB peak memory
- Required custom MLX fork with 3× Metal buffer limit (499K → 1.5M)
- Val loss 0.497, train loss 0.177 at iter 270

### 3. First Qwen3.5-122B-A10B Opus-Distilled Model
No 122B Opus-distilled model exists on HuggingFace. Jackrong published 9B, 27B, and 35B variants — we created the first 122B.

- Distilled from Claude Opus 4.6 reasoning traces (11,880 examples)
- 5-phase training pipeline: SFT curriculum → SimPO → GRPO → merge → GGUF

### 4. SpikingKiki — first 35B MoE → SNN conversion
LoRA-to-LAS rate-coded conversion of Qwen3.6-35B-A3B (31070 layers, 128 timesteps) — `convert_lora_to_snn.py` + `convert_spikingkiki_35b.py`.

## Micro_KIKI — 32 Expert Fleet

Fleet of 32 specialized MoE-LoRA experts on Qwen3.5-4B using [Brainstacks](https://arxiv.org/abs/2604.01152) (null-space projection for zero-forgetting continual learning). Deployable on RTX 4090 24 GB or Mac Studio.

**Domains:** 12 coding languages + 10 embedded/hardware + 10 general (reasoning, French, web, etc.)

```bash
# Data pipeline (1.57M raw → 63K deduplicated)
bash scripts/micro_kiki/pipeline_data.sh

# Train all 32 stacks sequentially (rank dynamique sqrt(N)/4 clampé [8,64])
bash scripts/micro_kiki/train_all_stacks.sh

# Evaluate forgetting matrix
uv run python scripts/micro_kiki/eval_stack.py --all
```

| Phase | Domains | Status (v4-sota) |
|-------|---------|--------|
| 1. Foundations | chat-fr, reasoning | Trained, **not yet evaluated** |
| 2. Coding core | python, typescript, cpp, rust (+ 4 v2 variants) | Trained, **not yet evaluated** |
| 3. Coding secondary | html-css, shell, sql, yaml-json, docker, kicad-dsl, spice, lua-upy | Trained, **not yet evaluated** |
| 4. Technical | embedded, stm32, iot, freecad, platformio, power, emc, dsp, electronics, kicad-pcb | Trained, **not yet evaluated** |
| 5. Applications | web-frontend, web-backend, music-audio, devops, llm-orch | Trained, **not yet evaluated** |
| 6. Complements | math, security | Trained, **not yet evaluated** |
| 7. ML Ops | llm-ops, ml-training | Trained (added late) |

> **Status note** : 39 adapters in `lora-qwen36-35b-v4-sota/` (Apr 19-26, ~130 GB). Last comparative eval is `bench-complete.json` (Apr 19), antérieur à la fin de la formation v4-sota. Lancer un nouveau bench est P1 dans la roadmap.

**Architecture:** 4 experts/stack, rank dynamique (sqrt(domain_size)/4 ∈ [8, 64]), top-k softmax routing on **all** experts (differentiable), rsLoRA scaling. Null-space projector now full `P = I - VᵀV` (was `V_keep` rank-k). ~250 MB per frozen stack, ~8 GB total for 32 stacks.

### 35B Brainstacks coexistence (3-phase curriculum)

`configs/mlx-lm-micro-kiki-phase{1,2,3}.yaml` — Qwen3.5-35B-A3B-Opus-bf16 with LoRA r64/α64, shared adapter `output/micro-kiki/stack-01-chat-fr`:

| Param | Phase 1 (foundations) | Phase 2 (medium) | Phase 3 (long) |
|---|---|---|---|
| `max_seq_length` | 512 | 1280 | **4096** |
| `batch_size` | 1 | **2** | 2 |
| `iters` | 500 | **1000** | 500 |
| `learning_rate` | 8e-6 | 5e-6 | **3e-6** |
| `grad_accumulation_steps` | 16 | 8 | 8 |

Curriculum : sequence extended progressively, LR decreasing, batch raised to 2 once Brainstacks complete (phase 2 comment: "batch=2 safe now"). Phase 3 adds 4k context support.

### Meta-router

`configs/micro-kiki-router.yaml` — Qwen3.5-4B frozen base, hidden=3072, MLP hidden=512, **top-k=4**, `chat_floor=0.20`, `gate_threshold=0.12`, attention pooling. Trained via `scripts/train_router.py` + `scripts/train_vqc_router.py` (variational quantum classifier comparison).

## ailiance — EU-sovereign sister stack

Sister project (`~/Documents/Projets/ailiance/`) using only EU/Swiss-origin models with full EU AI Act Article 52/53 transparency.

| Model | Origin | Domains | Config |
|-------|--------|---------|--------|
| Apertus 70B Instruct | EPFL+ETH+CSCS (CH) | 20 (electronics, EMC, DSP, SPICE, KiCad, STM32, IoT, embedded, MISRA-C, AUTOSAR, IEC norms…) | `configs/ailiance-apertus-electronics.yaml` |
| Devstral 2 24B MLX-4bit | Mistral AI (FR) | 16 (Python, Rust, TS, C++, shell, SQL, web, Docker, devops, llm-ops, ml-training…) | `configs/ailiance-devstral-python.yaml` |
| EuroLLM 22B Instruct | utter-project (EU) | 4 (chat-fr, traduction-tech, redaction-multilingue, localisation-doc) | `configs/ailiance-eurollm-chatfr.yaml` |

All 3 trained via `scripts/train_eu_kiki_{apertus,devstral,eurollm}.py` + sequential `train_eu_kiki_batch.py` and HF-traceable `train_eu_kiki_hf_batch.py`. LoRA r16/α32 on `q/k/v/o_proj`, all-linear bf16. See `ailiance/docs/eu-ai-act-transparency.md` for full provenance.

## Sonnet-Devstral Pipeline

Fine-tune Devstral 2 123B (dense, 72.2% SWE-bench) for fast Sonnet-style coding. Mixed dataset ~18K: R1 reasoning traces, code instructions, agentic SWE trajectories. Target languages: Python, TypeScript, Rust, Go.

```bash
./scripts/download_devstral.sh datasets   # download coding datasets
python scripts/prepare_coding_dataset.py  # build 18K filtered examples
./scripts/download_devstral.sh model      # download Devstral 2 123B (~250 GB)
python scripts/train_devstral_sonnet.py   # launch LoRA training
```

Config: `configs/mlx-lm-devstral2-sonnet.yaml` — LoRA rank 64, 4096 seq, 2000 iters.

## Benchmark suite

16 scripts (≈5300 lines) under `scripts/{bench_,eval_,benchmark_,test_}*.py` and `scripts/micro_kiki/eval_stack.py`.

### Réellement opérationnels

| Script | Quoi | Sortie |
|--------|------|--------|
| `bench_full.py` | Perplexité 35 domaines, base vs LoRA 35B vs LoRA 4B | `output/micro-kiki/eval/bench-35b-vs-4b.json` |
| `bench_complete.py` | 5 métriques × 35 domaines (val_ppl 25 samples, keyword_rate, response_len, degenerate_pct, optional 480B judge sur `--judge-url`) | `output/micro-kiki/eval/bench-complete.json` |
| `benchmark_base_models.py` | Compare modèles base (Qwen3.6 vs Granite, etc.) — perplexity + tok/s | `output/micro-kiki/eval/base_model_comparison.json` |
| `eval_v2_v3.py` (1512 l) | V2 vs V3 stacks, score composite 40% ppl + 40% kw + 20% length, matrice forgetting cross-eval | `results/v2-vs-v3.json` |
| `eval_aeon.py` | Recall@1/5/10 mémoire AeonPalace (100 épisodes synthétiques) | `results/aeon-eval.json` |
| `micro_kiki/eval_stack.py` | **Forgetting matrix** — pour chaque stack v3, déwrappe MoE-LoRA précédent puis évalue sur tous les autres domaines | stdout + JSON inline |
| `test_runtime_real.py` | Smoke test : adapter health (zero/nonzero LoRA-B), logit stats base vs +adapter | stdout |
| `test_gguf_domains.py` | POST 10 prompts vers `llama-server :8080`, tokens/sec + degenerate detection | `output/micro-kiki/gguf/smoke-test-results.json` |

### Scaffolds (renvoient des scores fake — NE PAS UTILISER tel quel)

> ⚠️ Ces scripts sont documentés "SCAFFOLD — stubbed inference" dans leur source. Marqués pour correction P1.

| Script | Stub à |
|--------|--------|
| `benchmark_base_vs_lora.py` | `_infer_stub` ligne 659, `_judge_stub` ligne 677 |
| `eval_niche_vs_base.py` | inférence stubbée ligne 397 (juge HTTP `localhost:8481` câblé mais inférence fake) |
| `eval_base_knowbias.py` | `compute_perplexity_mock` ligne 61-79 (hash-based) |
| `group_eval.py` | framework-only, pas d'inférence |

### Jamais exécutés

- `benchmark_quantum_router.py` — pas de fichier sortie
- `energy_bench.py` — calcul théorique pur (FLOPs ANN vs SNN), pas de sortie persistée
- `~/ailiance/scripts/eval_framework.py` (52 ko, EU AI Act Art. 53(1)(d)) — `output/eval/raw/` vide

### Pas de framework central

Pas de Makefile ni CI. 3 wrappers shell :
- `scripts/run_full_eval.sh` — 3/4 étapes sont des `echo` placeholders
- `scripts/run_forgetting.sh` — appelle `python -m src.eval.forgetting` avec stack-id
- `~/ailiance/scripts/run_eval.sh` — le plus propre (pre-flight checks, parsing CLI)

> 📋 **Lacunes** : pas de pass@1 / HumanEval / MBPP / GSM8K / MMLU-Pro câblés. Pas de versionning des résultats (écrasés à chaque run). Pas de standard de chemin de sortie (`results/` vs `output/micro-kiki/eval/` vs `~/ailiance/output/eval/raw/`).

## Cognitive runtime modules

`src/cognitive/` (non-versioned, in-progress) — judge, antibias, forgetting-gate, sleep-tagger, RBD, catfish. `src/serving/` — `mlx_server`, `ane_router`, `moe_lora_runtime`. `src/stacks/` — `oplora`, `qtha`, `moe_lora`, `trainer`. `src/compress/compactifai.py`.

## Models

| Model | Size | Location |
|-------|------|----------|
| Qwen3.6-35B-A3B-MLX-BF16 | ~65 GB | `models/` |
| Mistral-Medium-3.5-128B-BF16 | ~240 GB | `models/` |
| Mistral-Medium-3.5-128B-MLX-Q8 | ~130 GB | `models/` |
| Qwen3.5-122B-A10B-BF16 | 233 GB | `models/` |
| Qwen3.5-35B-A3B-Opus-bf16 | 65 GB | `models/` |
| Qwen3.5-35B-A3B-Opus-vlm | — | fusion model (vision tower) |
| Mistral Large 123B | ~250 GB | `models/` |
| Devstral 2 123B (dense) | ~250 GB | `models/` |
| Apertus 70B Instruct | ~140 GB | `models/` |
| EuroLLM 22B Instruct | ~44 GB | `models/` |

## Architecture

```
ailiance-mac-tuner/
├── setup.sh / download.sh / train.sh / export.sh   # main workflow
├── configs/                       # training + generation YAML configs
│   ├── ailiance-{apertus,devstral,eurollm}-*.yaml   # EU-sovereign training
│   ├── mlx-lm-micro-kiki-phase{1,2,3}.yaml         # 35B Brainstacks curriculum
│   ├── micro-kiki-router.yaml                      # 32-domain meta-router
│   └── micro_kiki/brainstacks.yaml                 # 32-stack fleet (Phase 7 added)
├── scripts/
│   ├── train_122b_macport.sh                       # 122B MoE training wrapper
│   ├── train_devstral_sonnet.py                    # Devstral 2 123B LoRA
│   ├── train_eu_kiki_*.py                          # EU-sovereign training
│   ├── micro_kiki/                                 # Brainstacks stack training
│   ├── bench_*.py / eval_*.py                      # 12+ benchmark scripts
│   ├── convert_lora_to_snn.py                      # SpikingKiki conversion
│   ├── build_hybrid_adapters.py                    # V2/V3 best-of selection
│   └── watchdog_mem.sh                             # swap-thrash kill switch
├── tools/
│   └── train_monitor_tui.py                        # live Rich TUI monitor
├── src/
│   ├── cognitive/                                  # judge, antibias, forgetting-gate…
│   ├── serving/                                    # mlx_server, ane_router, moe_lora_runtime
│   ├── stacks/                                     # oplora, qtha, moe_lora, trainer
│   └── compress/compactifai.py
├── data/                                           # datasets (downloaded)
├── output/                                         # checkpoints + LoRA adapters
├── models/                                         # downloaded base models
├── results/                                        # eval result JSONs (v2-vs-v3, spikingkiki…)
├── lib/
│   └── mlx_lm_fork/                                # SSD offload for MoE experts
├── research/
│   └── ane-hybrid/                                 # ANE + CoreML pipeline research
└── docs/
    ├── plans/                                      # 10 implementation plans (122B, devstral-v4, ANE…)
    ├── specs/                                      # micro-kiki-design.md
    ├── sota-training-2026.md                       # Apple Silicon SOTA techniques
    └── micro-kiki-moe-research.md                  # 32 LoRA experts on RTX 4090
```

## Key Dependencies

- [MLX](https://github.com/ml-explore/mlx) (custom fork at `/tmp/mlx-fork` with 3× Metal buffer limit)
- [mlx-lm](https://github.com/ml-explore/mlx-lm) (with vendored `lib/mlx_lm_fork/` for SSD offload)
- [mlx-tune](https://github.com/ml-explore/mlx-tune) ≥ 0.4.21
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) 0.4.4 — vision-language model training and inference on Apple Silicon

## License

MIT
