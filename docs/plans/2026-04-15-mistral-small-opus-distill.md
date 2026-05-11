# Mistral-Small-24B Opus distillation pipeline

Date: 2026-04-15
Author: claude + clement
Status: **PLAN** (do not execute until Devstral bench + Mistral bench chain done, ~01:00 tomorrow)

---

## 1. Objectif

Distiller **Mistral-Large-Opus** (teacher, ~123 B BF16, fused sur Studio) dans **Mistral-Small-3.1-24B-Instruct-2503** (student), puis exporter en GGUF Q4_K_M **~14 GB** qui tient entièrement sur le RTX 4090 (24 GB VRAM) de kxkm-ai. Cible qualité : **≥ 90 %** du teacher sur `run-full-bench.sh` (120 FR prompts + eval_ab).

Pipeline inspiré de Devstral v3 (SFT + DAPO + fuse + GGUF), qui a donné 10/10 think + 10/10 code, 3.52 s avg.

---

## 2. Architecture

| Rôle | Modèle | Machine | Quant |
|---|---|---|---|
| Teacher | `mistral-large-opus-fused` (`/Users/clems/ailiance-mac-tuner/output/mistral-large-opus-fused/`) | Studio M3 Ultra 512 GB | BF16, ~233 GB |
| Student base | `unsloth/mistral-small-3.1-24b-instruct-2503` (déjà dans HF cache kxkm-ai) | kxkm-ai RTX 4090 | 4bit bnb pour SFT, BF16 pour fuse |
| Student final | `mistral-small-opus-Q4_K_M.gguf` | déployable 4090 | Q4_K_M, ~14 GB |

Pattern : QLoRA 4bit via Unsloth (comme `train_sft_v2.py` pour Devstral). Rank 64, alpha 64, grad_checkpoint unsloth.

---

## 3. Pipeline (4 phases)

### Phase 1 — Dataset generation (teacher forward, ~12-18 h)

Deux sources à fusionner (~15-20 k paires `{user, assistant}`) :

1. **Prompts existants** : sous-échantillon de `mega-v2` (498 k examples en 25 domaines, Tower `~/mega-dataset/`) — tirer ~12 k prompts stratifiés par domaine.
2. **Prompts eval-proches** : augmenter `eval_prompts.jsonl` (120) via paraphrase → ~3 k FR prompts diversifiés.

Teacher sert via `llama-server` :
- **Option A (préférée)** : lancer `llama-server` sur **Studio** après la bench chain (RAM OK, teacher déjà présent), expose sur LAN.
- **Option B (fallback)** : `mlx_lm.server` sur Studio (déjà testé pour fine-tune 35B).

Appel depuis kxkm-ai via autossh tunnel `localhost:18000 → studio:8000` (même pattern que le tunnel kxkm-ai ↔ Tower). Script : `scripts/generate-mistral-large-distill.py` (resume-safe, append-only).

Sortie : `/home/kxkm/kiki-v3/data/mistral-large-opus-distill.jsonl`. Split 95/5 train/valid via ligne `head/tail` ou script existant.

### Phase 2 — SFT student (~10 h sur 4090)

Réutiliser `train_sft_v2.py` avec `--base-model unsloth/mistral-small-3.1-24b-instruct-2503`. Config : `configs/mistral-small-opus-sft.yaml`.

- 2 epochs, lr 1e-5, cosine, warmup 100, max_seq 4096
- rank 64 / alpha 64 / dropout 0.01 / 7 target_modules
- grad_accum 8 → effective batch 8
- 4bit QLoRA Unsloth — ~18 GB VRAM pic

Artefact : `outputs/mistral-small-opus-sft/` (LoRA adapter).

### Phase 3 — DAPO optionnel (~6 h)

Réutiliser `train_dapo.py` + `hard_reward.py`. Preference pairs générées par teacher-as-judge sur ~2 k prompts (`scripts/generate-mistral-large-distill.py` en mode judge à écrire plus tard).

- Skip si SFT seul atteint déjà ≥90 % du teacher.
- Sinon : 100-150 GRPO steps max, même rank/alpha que SFT, chargement via `FastLanguageModel.from_pretrained(outputs/mistral-small-opus-sft, load_in_4bit=True)`.

### Phase 4 — Fuse + GGUF + deploy (~1 h)

1. Merge LoRA dans base BF16 → `outputs/mistral-small-opus-merged/` (via `unsloth.save_pretrained_merged` ou script Mac-side).
2. Convert → GGUF avec `llama.cpp/convert_hf_to_gguf.py` (F16 intermédiaire optionnel).
3. Quantize `llama-quantize merged.gguf mistral-small-opus-Q4_K_M.gguf Q4_K_M` → **~14 GB**.
4. Déployer sur 4090 via `llm-swap.sh` (voir §6).

---

## 4. Ressources & timing

| Phase | Machine | Durée | Bloque quoi ? |
|---|---|---|---|
| 1 Dataset gen | Studio (teacher) + kxkm-ai (client) | ~12-18 h | Studio CPU/GPU, pas d'autre SFT possible en // |
| 2 SFT | kxkm-ai RTX 4090 | ~10 h | 4090 mono-job |
| 3 DAPO (opt) | kxkm-ai RTX 4090 | ~6 h | idem |
| 4 Fuse + GGUF | kxkm-ai (ou Studio mlx→gguf) | ~1 h | léger |
| **Total** | | **~30 h** (SFT only) / **~36 h** (+ DAPO) | |

Dépendances amont : **attendre fin de Devstral bench + rerun + Mistral bench chain** (~01:00 2026-04-16) avant de kill tout llama-server et basculer teacher.

---

## 5. Configs (skeletons)

- **SFT** : `/home/kxkm/kiki-v3/configs/mistral-small-opus-sft.yaml` — voir le fichier ; parse OK.
- **DAPO** : à dériver de l'appel `train_dapo.py` Devstral v3 (même `hard_reward.hybrid_reward`, juste swap `--resume-adapter` et `--prompts`).

---

## 6. Deploy

### systemd unit template (kxkm-ai, `/etc/systemd/system/llama-mistral-small-opus.service`)

```ini
[Unit]
Description=llama-server Mistral-Small-Opus Q4_K_M
After=network.target

[Service]
Type=simple
User=kxkm
WorkingDirectory=/home/kxkm/kiki-v3/outputs
ExecStart=/home/kxkm/llama.cpp/build/bin/llama-server \
  -m /home/kxkm/kiki-v3/outputs/mistral-small-opus-Q4_K_M.gguf \
  -c 8192 -ngl 99 --host 0.0.0.0 --port 8000 \
  --chat-template mistral
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

### llm-swap.sh integration

Ajouter une entrée dans `/home/kxkm/infra/bin/llm-swap.sh` :

```bash
mistral-small-opus)
  MODEL=/home/kxkm/kiki-v3/outputs/mistral-small-opus-Q4_K_M.gguf
  CTX=8192 ; NGL=99 ; TEMPLATE=mistral
  ;;
```

Puis `llm-swap.sh mistral-small-opus` pour hot-swap.

---

## 7. Eval

Réutiliser sans modification :
- `run-full-bench.sh` (120 prompts FR de `eval_prompts.jsonl`)
- `/home/kxkm/infra/bin/llm-eval.py` pour scoring
- `eval_ab.py` pour A/B vs Mistral-Large-Opus (teacher en ref)

Gate qualité : score_student / score_teacher ≥ 0.90 sur bench complet, sinon passer Phase 3 (DAPO) ou relancer SFT avec rank 96 / 3 epochs.

---

## 8. Risques

| Risque | Impact | Mitigation |
|---|---|---|
| Capacity loss 123B→24B : certains domaines bas perf | Qualité | Augmenter rank (64→96) ; 3 epochs au lieu de 2 ; DAPO activé |
| Q4_K_M degradation (esp. reasoning FR) | Qualité deploy | Garder aussi Q5_K_M fallback (~17 GB tient encore sur 4090) |
| Dataset bias (mega-v2 surreprésente code) | Qualité FR conversation | Stratifier par domaine, boost domaines FR littéraire/math/eval-like |
| OOM 4090 sur seq_len 4096 | Blocage SFT | Fallback seq 2048 + grad_accum 16 (commenté dans YAML) |
| Teacher down mid-gen | Dataset incomplet | Script resume-safe (ids déjà faits skippés) |
| Clash avec bench chain en cours | Corrompt bench | Ne rien démarrer avant 01:00 tomorrow |
| Unsloth/bnb API breaking change | Train fail | Pinner versions de l'environnement Devstral v3 (`~/kiki-v3` venv) |

---

## 9. Next steps (commandes à déclencher)

Ordre strict, **après** la fin de la bench chain (~2026-04-16 01:00+).

```bash
# 0. Vérifier que rien ne tourne
ssh kxkm@kxkm-ai "nvidia-smi ; pgrep -a llama-server ; pgrep -af train_"

# 1. Lancer teacher sur Studio (dans un tmux)
ssh studio "tmux new -d -s teacher 'cd ~/llama.cpp && ./build/bin/llama-server \
  -m /Users/clems/ailiance-mac-tuner/output/mistral-large-opus-fused/ggml-model-BF16.gguf \
  -c 8192 -ngl 99 --host 0.0.0.0 --port 8000'"
# (si fusion n'a pas encore de GGUF : passer par mlx_lm.server + openai adapter)

# 2. Tunnel kxkm-ai -> Studio
ssh kxkm@kxkm-ai "autossh -M 0 -f -N -L 18000:localhost:8000 studio"

# 3. Préparer prompts distill sur kxkm-ai
ssh kxkm@kxkm-ai "cd ~/kiki-v3 && \
  python scripts/sample_mega_prompts.py --n 12000 --out data/distill_prompts.jsonl && \
  cat /Users/clems/ailiance-mac-tuner/configs/eval_prompts.jsonl >> data/distill_prompts.jsonl"
# (sample_mega_prompts.py reste à écrire, ~30 lignes)

# 4. Phase 1 — teacher forward (background, ~12-18 h)
ssh kxkm@kxkm-ai "tmux new -d -s distill-gen 'cd ~/kiki-v3 && \
  python scripts/generate-mistral-large-distill.py \
    --prompts-file data/distill_prompts.jsonl \
    --teacher-url http://localhost:18000/v1/chat/completions \
    --out data/mistral-large-opus-distill.jsonl \
    --model mistral-large-opus --max-tokens 2048 --temperature 0.3'"

# 5. Split train/valid
ssh kxkm@kxkm-ai "cd ~/kiki-v3 && \
  shuf data/mistral-large-opus-distill.jsonl > /tmp/sh.jsonl && \
  split -l $(( $(wc -l < /tmp/sh.jsonl) * 95 / 100 )) /tmp/sh.jsonl /tmp/mls_ && \
  mv /tmp/mls_aa data/mistral-large-opus-distill.train.jsonl && \
  mv /tmp/mls_ab data/mistral-large-opus-distill.valid.jsonl"

# 6. Phase 2 — SFT (~10 h)
ssh kxkm@kxkm-ai "tmux new -d -s mls-sft 'cd ~/kiki-v3 && \
  python train_sft_v2.py \
    --train data/mistral-large-opus-distill.train.jsonl \
    --valid data/mistral-large-opus-distill.valid.jsonl \
    --output outputs/mistral-small-opus-sft \
    --epochs 2 --lr 1e-5 --rank 64 --grad-accum 8 --max-seq 4096'"
# NB: si train_sft_v2.py n'accepte pas --base-model, patcher la ligne model_name
#     en tête de script, ou dupliquer en train_sft_v2_mls.py.

# 7. (optionnel) Phase 3 — DAPO
# voir train_dapo.py avec --resume-adapter outputs/mistral-small-opus-sft

# 8. Phase 4 — fuse + GGUF + deploy
ssh kxkm@kxkm-ai "cd ~/kiki-v3 && python scripts/fuse_and_quantize.py \
  --adapter outputs/mistral-small-opus-sft \
  --base unsloth/mistral-small-3.1-24b-instruct-2503 \
  --out outputs/mistral-small-opus-Q4_K_M.gguf --qtype Q4_K_M"
# puis llm-swap.sh mistral-small-opus && run-full-bench.sh
```

---

## Annexes

- Artefacts créés par ce plan :
  - `/home/kxkm/kiki-v3/configs/mistral-small-opus-sft.yaml` (2.4 KB, YAML OK)
  - `/home/kxkm/kiki-v3/scripts/generate-mistral-large-distill.py` (6.4 KB, 180 lignes, py_compile OK)
  - ce doc : `/Users/clems/ailiance-mac-tuner/docs/plans/2026-04-15-mistral-small-opus-distill.md`
- À écrire au moment de lancer :
  - `scripts/sample_mega_prompts.py` (~30 lignes, stratified sampler)
  - `scripts/fuse_and_quantize.py` (~50 lignes, wraps unsloth merge + llama.cpp convert + quantize)
  - train_dapo config si DAPO activé

## §10 Améliorations intégrées (PiSSA + DoRA + SimPO)

**Pipeline révisé** :

| Phase | Tech standard | Tech améliorée | Gain |
|-------|---------------|-----------------|------|
| SFT | LoRA rank 64 | **PiSSA + DoRA** | Convergence 2×, qualité +1-2 % |
| Alignment | DAPO (GRPO) | **SimPO** | Économise 1 reference model (~24 GB RAM), qualité ≈ DPO |
| RL (optionnel) | — | **RLTT** | +14 % MATH vs GRPO, plus complexe, à tester après |

### PiSSA
Init des matrices LoRA via SVD des poids de base au lieu de random. Drop-in Unsloth : `init_lora_weights: pissa` + `pissa_niter: 4`. Convergence 2× plus rapide → on peut couper les epochs de 2 à 1-1.5.

### DoRA
Décompose W = magnitude × direction, LoRA uniquement sur la direction. Active via `use_dora: true`. Overhead compute ~15 %, gain qualité 1-2 % sur benchs standard.

### SimPO
Reference-free DPO. Remplace la phase alignment DAPO/GRPO par SimPO via TRL DPOTrainer :
- `loss_type: simpo`
- `beta: 2.0` (reward scale)
- `gamma_beta_ratio: 0.5` (target reward margin)
Script dédié : `kiki-v3/train_simpo.py`. Config : `configs/mistral-small-opus-simpo.yaml`.

### Configs affectés
- `configs/mistral-small-opus-sft.yaml` : + `init_lora_weights: pissa`, + `use_dora: true`, + `pissa_niter: 4`
- `configs/mistral-small-opus-simpo.yaml` : **nouveau** (remplace l'idée DAPO initiale)
- `train_simpo.py` : **nouveau** sur kxkm-ai

### Ordonnancement commande
```bash
# Phase 2 (SFT) avec PiSSA + DoRA
ssh kxkm@kxkm-ai "cd /home/kxkm/kiki-v3 && /home/kxkm/.venv/bin/python train_sft_v2.py configs/mistral-small-opus-sft.yaml"

# Phase 3 (SimPO)
ssh kxkm@kxkm-ai "cd /home/kxkm/kiki-v3 && /home/kxkm/.venv/bin/python train_simpo.py configs/mistral-small-opus-simpo.yaml"
```

### ETA révisé avec ces optimisations
- SFT : ~10 h → **~5-6 h** (PiSSA 2× + DoRA -15 % offsets partly)
- Alignment : ~6 h → **~3-4 h** (SimPO sans ref model)
- Fuse + GGUF : ~1 h
- **Total : ~10-12 h** (vs ~18 h pipeline standard)
