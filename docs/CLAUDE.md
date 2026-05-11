# Documentation

## Plans

| Plan | Date | Statut |
|------|------|--------|
| Fix OOM training | 2026-04-12 | Fait |
| ANE hybrid pipeline | 2026-04-14 | Phases 1-3 faites |
| 122B Opus-v3 training | 2026-04-15 | Fait (val 0.468) |
| Qwen3.5-122B mac-port | 2026-04-15 | Fait |
| Mistral Small distill | 2026-04-15 | Fait |
| Devstral v4 Opus distill | 2026-04-15 | En cours (ailiance devstral-python) |
| Micro_KIKI plan 1 (data) | 2026-04-15 | Fait (63K examples, 32 domaines) |
| Micro_KIKI plan 2 (Brainstacks) | 2026-04-15 | **Fait** (32 piles entraînées, rank dynamique) |
| Micro_KIKI plan 3 (meta-router) | 2026-04-15 | Config + scripts en place |
| Micro_KIKI plan 4 (ANE pipeline) | 2026-04-15 | En cours |

## Datasets disponibles

| Source | Exemples |
|--------|----------|
| Opus 3K original | 2326 |
| Opus 12K SFT | 11673 |
| Combined-opus-14k (dédupliqué) | 9813 |
| Distillé 123B | 87 |
| Distillé 35B batch 1+2 | ~2000 |
| Distillé mlx-vlm | ~150 |
| **final-opus-v3-1** | **11880 train + 626 valid** |
| Brainstacks 32 domaines | 1.57M raw → 63K dédupliqué |
| Devstral-Sonnet (R1 + SWE) | ~18K |

## Modèles entraînés

| Modèle | Val loss | Train loss | Checkpoint | Statut |
|--------|----------|------------|------------|--------|
| Mistral Large 123B LoRA | 0.479 | — | iter 1100 | Terminé |
| Qwen3.5-122B-A10B Opus v3 | 0.468 | 0.177 (iter 270) | iter 400 | Terminé |
| Qwen3.5-35B-A3B Opus v3 | — | — | published | Terminé |
| Brainstacks fleet (32 piles) | — | — | output/micro-kiki/stack-*/  | Terminé |
| SpikingKiki-35B-V4 | — | — | results/spikingkiki-35b-convert.json | Conversion faite (31070 layers, 128 timesteps) |

## Résultats d'éval (`results/`)

| Fichier | Date | Verdict |
|---------|------|---------|
| `v2-vs-v3.json` | 2026-04-17 | V3 vs V2 sur 10 niches : 0 wins V3 / 1 win V2 / 9 ties — `build_hybrid_adapters.py` sélectionne best-of par domaine. |
| `spikingkiki-35b-convert.json` | 2026-04-22 | Qwen3.6-35B-A3B → SpikingKiki-35B-V4 OK (11086 s). |

## Benchmarks inference

| Modèle | Moteur | tok/s |
|--------|--------|-------|
| Qwen3.5-35B-A3B | mlx-vlm natif | 45-89 |
| DeltaNet 40 couches ANE | CoreML | 14.4 (474/couche) |
| MLX pur (modèle complet) | MLX | 14.2 |

## Pipeline Sonnet-Devstral

| Fichier | Rôle |
|---------|------|
| `configs/mlx-lm-devstral2-sonnet.yaml` | Config LoRA pour Devstral 2 123B coding |
| `scripts/download_devstral.sh` | Téléchargement modèle + 7 datasets coding |
| `scripts/prepare_coding_dataset.py` | Fusion, filtrage, dédup → 18K exemples |
| `scripts/train_devstral_sonnet.py` | Training mlx-tune LoRA sur Devstral 2 dense |
| `data/sonnet-coding/` | Dataset final (train.jsonl + valid.jsonl) |

Datasets sources : OpenCodeReasoning (nvidia), OpenCodeInstruct (nvidia), Codeforces-CoTs (open-r1), Magicoder OSS-Instruct, CodeFeedback, OpenHands trajectoires, Nemotron-SWE.

## Pipeline ailiance (sister project)

3 modèles EU-souverains entraînés en parallèle :

| Modèle | Origine | Domaines | Config |
|--------|---------|----------|--------|
| Apertus 70B Instruct | EPFL+ETH+CSCS (CH) | 20 (electronics, EMC, DSP, SPICE, KiCad, IEC norms…) | `configs/ailiance-apertus-electronics.yaml` |
| Devstral 2 24B MLX-4bit | Mistral AI (FR) | 16 (Python, Rust, TS, C++, devops, llm-ops, ml-training…) | `configs/ailiance-devstral-python.yaml` |
| EuroLLM 22B Instruct | utter-project (EU) | 4 (chat-fr, traduction-tech, redaction-multilingue) | `configs/ailiance-eurollm-chatfr.yaml` |

Scripts : `scripts/train_eu_kiki_{apertus,devstral,eurollm}.py`, batch séquentiel `scripts/train_eu_kiki_batch.py`, HF-traceable `scripts/train_eu_kiki_hf_batch.py`. Voir aussi `~/Documents/Projets/ailiance/`.

## Micro_KIKI Pipeline

32 experts MoE-LoRA sur Qwen3.5-4B via Brainstacks (null-space projection + residual boosting).

| Étape | Fichiers | Statut |
|-------|----------|--------|
| Data pipeline | `scripts/micro_kiki/classify_parallel.py`, `deduplicate.py`, `split_domains.py` | Fait (63K exemples, 32 domaines) |
| Brainstacks training | `scripts/micro_kiki/train_stack.py` (rank dynamique sqrt(N)/4 ∈ [8,64]), `eval_stack.py`, `train_all_stacks.sh` | **Fait** (32 piles entraînées) |
| 35B coexistence curriculum | `configs/mlx-lm-micro-kiki-phase{1,2,3}.yaml` | En place (seq 512 → 1280 → 4096) |
| Config | `configs/micro_kiki/brainstacks.yaml` (Phase 7 ajoutée), `domains.yaml` | Fait |
| Meta-router | `configs/micro-kiki-router.yaml`, `scripts/train_router.py`, `train_vqc_router.py` | Config + scripts en place |
| Plans | `docs/plans/2026-04-15-micro-kiki-plan{1-4}*.md` | Plans 1-4 écrits |

Datasets : 1.57M raw → 63K dedup (25 sources : CodeFeedback, Glaive-v3, OASST2, French-Alpaca-110K, Trendyol-Cybersec, STM32-HAL, LTspice, etc.)

## Points d'attention Brainstacks

- **Rank dynamique** : `sqrt(N_domain)/4` clampé `[8, 64]` (`scripts/micro_kiki/train_stack.py`)
- **Routing top-k différentiable** : softmax sur **tous** les experts (fix `mx.topk` non-différentiable)
- **Null-space projector complet** : `P = I - VᵀV` (était `V_keep` rank-k)
- **Suffix matching gradients/projecteurs** : strip `language_model.model.` + guards dimensions (`residual_boost.py`)
- **Phase 7 ajoutée** : `llm-ops`, `ml-training`

## Suite SpikingKiki

| Fichier | Rôle |
|---------|------|
| `scripts/convert_lora_to_snn.py` | Conversion LoRA → LAS rate-coded |
| `scripts/convert_spikingkiki_35b.py` | Conversion 35B-A3B → SpikingKiki-35B-V4 |
| `scripts/quantize_spikingbrain.py` | Quantization Q4 |
| `scripts/probe_spikingbrain_hf.py`, `smoke_spikingbrain.py`, `validate_spiking_e.py` | Validation E2E |
| `scripts/fork_qwen_diffattn.py` | Patch DiffAttn pour Qwen |

## Modules `src/` (runtime cognitif, en cours)

| Module | Contenu |
|--------|---------|
| `src/cognitive/` | judge, antibias, forgetting-gate, sleep-tagger, RBD, catfish |
| `src/serving/` | mlx_server, ane_router, moe_lora_runtime |
| `src/stacks/` | oplora, qtha, moe_lora, trainer |
| `src/compress/` | compactifai.py |

## Infra clé

- mlx-tune 0.4.21+
- MLX fork 3x Metal limit (`/tmp/mlx-fork`) installé dans venv
- `lib/mlx_lm_fork/` — fork mlx_lm avec SSD offload pour MoE experts
- llama.cpp CPU+GPU compilé dans `/tmp/`
- ANEMLL dans `/tmp/anemll`
- Peak mem training 122B : 383 Go
- `iogpu.wired_limit_mb=458752` requis avant 122B
