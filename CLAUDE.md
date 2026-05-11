# ailiance-mac-tuner

Fine-tuning LLMs sur Apple Silicon (512 Go RAM unifiée) via MLX.
Distille le raisonnement Claude Opus dans des modèles open-source.

## Machine

Mac Studio M3 Ultra, 512 Go mémoire unifiée. MLX bf16 complet.

## Workflow principal

```
./setup.sh → ./download.sh → ./train.sh → ./export.sh
```

## Sous-projets actifs

| Track | Entrée | Modèles |
|-------|--------|---------|
| **Foundation distill** | `./train.sh`, `scripts/train_122b_macport.sh` | Mistral Large 123B, Qwen3.5-122B-A10B, Qwen3.5-35B-A3B |
| **Brainstacks 32-fleet** | `scripts/micro_kiki/train_all_stacks.sh` | Qwen3.5-4B (RTX 4090) + Qwen3.5-35B-A3B-Opus (curriculum 3 phases) |
| **ailiance** | `scripts/train_eu_kiki_batch.py` | Apertus 70B (CH), Devstral 2 24B (FR), EuroLLM 22B (EU) |
| **Sonnet-Devstral** | `scripts/train_devstral_sonnet.py` | Devstral 2 123B dense |
| **SpikingKiki** | `scripts/convert_lora_to_snn.py`, `convert_spikingkiki_35b.py` | Qwen3.6-35B-A3B → SNN LAS |
| **ANE hybrid** | `research/ane-hybrid/` | DeltaNet → CoreML/ANE |
| **Meta-router** | `configs/micro-kiki-router.yaml`, `scripts/train_router.py` | Qwen3.5-4B routeur 32 domaines |

## Where to Look

| Tâche | Emplacement |
|-------|-------------|
| Configs training/generation | `configs/` |
| Configs ailiance (3 modèles EU) | `configs/ailiance-*.yaml` |
| Configs micro-kiki 35B curriculum | `configs/mlx-lm-micro-kiki-phase{1,2,3}.yaml` |
| Config meta-router 32 domaines | `configs/micro-kiki-router.yaml` |
| Config Brainstacks 32 piles | `configs/micro_kiki/brainstacks.yaml` |
| Scripts foundation distill | `scripts/train_*.py`, `scripts/distill*.sh` |
| Scripts Brainstacks | `scripts/micro_kiki/` |
| Scripts ailiance | `scripts/train_eu_kiki_*.py` |
| Scripts SpikingBrain/SNN | `scripts/convert_*spiking*.py`, `scripts/quantize_spikingbrain.py` |
| Scripts bench/eval | `scripts/bench_*.py`, `scripts/eval_*.py` |
| Datasets | `data/` |
| Checkpoints et LoRA | `output/` |
| Résultats d'éval | `results/` (v2-vs-v3.json, spikingkiki-*.json…) |
| Recherche ANE hybrid | `research/ane-hybrid/` |
| Plans d'implémentation | `docs/plans/` (10 plans : 122B v3, devstral-v4, ANE, fix-OOM…) |
| Specs design | `docs/specs/` |
| Recherche SOTA / MoE | `docs/sota-training-2026.md`, `docs/micro-kiki-moe-research.md` |
| Modules runtime cognitif | `src/cognitive/`, `src/serving/`, `src/stacks/` |
| Fork mlx_lm (SSD offload) | `lib/mlx_lm_fork/` |
| Fork MLX (3x Metal limit) | `/tmp/mlx-fork` (installé dans venv) |
| Modèles téléchargés | `models/` |

## Dataset format

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "<thinking>...</thinking>\n\n..."}]}
```

## Brainstacks — points clés

- **Rank dynamique** : `sqrt(N_domain)/4` clampé `[8, 64]` (`scripts/micro_kiki/train_stack.py`)
- **Routing top-k différentiable** : softmax sur **tous** les experts (fix `mx.topk` non-différentiable)
- **Null-space projector complet** : `P = I - VᵀV` (était `V_keep` rank-k)
- **Phase 7 ajoutée** : `llm-ops`, `ml-training` (dans `brainstacks.yaml`)
- **Coexistence 35B** : 3 phases curriculum (seq 512 → 1280 → 4096, lr 8e-6 → 5e-6 → 3e-6, batch 1 → 2 → 2)

## Roadmap (audit 2026-05-04)

État réel d'entraînement vs annoncé : voir `docs/CLAUDE.md` "Modèles entraînés" et `docs/CLAUDE.md` "Résultats d'éval". Plusieurs trainings annoncés "Done" sont en réalité **CRASHED** ou **NOT STARTED**.

### 🔴 Bloquants (à traiter d'abord)

1. **Choisir version canonique brainstacks** — v3-r16 mort (lora_B=0), v4-dynrank partiel (31/32), v4-sota terminé Apr 26 (~130 GB) mais jamais évalué. Décider, archiver le reste, libérer ~150 GB. (1 demi-journée)
2. **Entraîner meta-router KIKI** (`output/micro-kiki/router/` vide) — sans lui, plans 3 + 4 morts. (~6-8 h compute)
3. **Décider sort 122B-Opus-v3** — `logs/curriculum.log` ligne 30 = OOM dès iter 1. Retry seq=896 + recompile MLX, ou abandon. (2-3 j si retry)
4. **Stabiliser ou abandonner VLM PoC Devstral** — loss diverge >5 sur 6 runs (`vlm_poc_run6.log`). Revoir prepro / lr / abandon. (1-2 j)
5. **Debug bug V2 ≡ V3 dans `eval_v2_v3.py`** — perplexités strictement identiques bit-à-bit dans `results/v2-vs-v3.json` et `output/micro-kiki/eval/fused_eval_results.json` → adapter probablement pas chargé. (0.5 j)

### 🟠 Important non-bloquant

- Lancer `~/ailiance/scripts/run_eval.sh --mode compare` (eval framework prêt mais `output/eval/raw/` vide). 3-5 h compute.
- Compléter Apertus (`emc-dsp-power`, `security-fenrir`) + EuroLLM (1 manquant). 12-15 h compute.
- Lancer Mistral-Small-Opus + Devstral v4 sur kxkm-ai (RTX 4090, plans-only à ce jour). 3-5 j.
- Quantization SpikingKiki Q4 + smoke test `smoke_spikingbrain.py` + energy bench réel. 1 j.
- Câbler des évals réelles : remplacer scaffolds (`benchmark_base_vs_lora.py:659,677`, `eval_niche_vs_base.py:397`, `eval_base_knowbias.py:61-79`) par HumanEval / MBPP / GSM8K. 1-2 j.
- Lancer `bench_complete.py` sur **v4-sota** terminé Apr 26 (dernier bench connu = Apr 19, antérieur). 2-3 h.

### 🟡 Cleanup

- Publier sur HF `mistral-large-opus` (3.36 GB, dormant) + stack AILIANCE v1 (Apertus/Devstral/EuroLLM) — pas encore sur `clemsail/` ni `electron-rare/`. `scripts/release_hf.py` prêt en dry-run, jamais lancé `--execute`. (3-4 h)
- Réconcilier les doublons HF : `clemsail/kiki-{stm32,kicad}-sft` (79+94 dl) vs `electron-rare/kiki-{stm32,kicad}-sft-v1` (0 dl). Archiver les v1 vides ou rapatrier les vrais. (1 h)
- Compléter les model cards des 5 modèles `clemsail/` à 0 dl (`micro-kiki-{v35b,router-v4,v4-sota}`, `spikingkiki-{35b-a3b-v4,v4-adapters}`). (2 h)
- Archiver dossiers fantômes (utiliser `tools/archive_dead_artifacts.sh`) :
  - `output/micro-kiki/stacks-v3-r16/` (14 GB, lora_B=0)
  - `output/micro-kiki/lora-qwen36-35b-hybrid/*` (dossiers vides)
  - `output/qwen35-122b-macport/`, `output/qwen35-35b-opus-14k-v1/`, `output/qwen35-35b-opus-final/` (configs seuls)
  - `output/micro-kiki/stack-01-chat-fr-v2/` (config seul)
- Standardiser convention output `results/<YYYY-MM-DD>-<scope>.json` (actuellement éparpillé sur 3 chemins).
- Documenter dans CLAUDE.md la convention v3/v4/v4-sota/dynrank.

### 🟢 Future / horizon long

- Phases ANE 2 (full stack) / 3 (MoE hybrid) / 3b (GPU experts) / 4 (full hybrid serving). Scripts présents dans `research/ane-hybrid/`, jamais exécutés. (~1 sem)
- Variant quantum router `benchmark_quantum_router.py` (jamais exécuté). 3-4 j.
- Pipeline VLM full (au-delà du PoC). 1 sem si convergence trouvée.

### Adapters orphelins (entraînés mais ni évalués ni publiés)

- 39 × `lora-qwen36-35b-v4-sota/*` (~130 GB, terminés Apr 26, jamais re-benchés)
- 32 × `stacks-v3-r16/*` (14 GB, effondrés, à supprimer)
- 31 × `~/ailiance/output/adapters/{apertus,devstral,eurollm}/*` (eval pas lancée)
- 1 × `output/mistral-large-opus/adapters.safetensors` (3.36 GB, dormant 21 jours, ni mergé ni publié)

## Anti-Patterns

- Ne pas fine-tuner en 4-bit — bf16 est gratuit avec 512 Go
- Ne pas utiliser PyTorch MPS — MLX est 3-5x plus rapide
- Ne pas oublier `--resume` après un Ctrl+C
- `huggingface-cli` deprecated → utiliser `hf`
- `mlx_lm.convert --dtype bf16` → `--dtype bfloat16`
- Pour le 122B : utiliser `mlx-tune` (0.4.21+), pas `mlx_lm.lora` directement
- MLX stock limite les Metal buffers à 499K → le fork 3x (`/tmp/mlx-fork`) est requis pour 122B bf16
- `iogpu.wired_limit_mb=458752` obligatoire avant training 122B (sinon OOM kernel)
- **Qwen3.5 thinking-mode trap** : `enable_thinking=false` obligatoire ou les scoreurs renvoient 0/N
- **Pré-pivot MoE-LoRA stacks-v3-r16** : morts (lora_B = 0), V4 sequential per-domain les remplace
