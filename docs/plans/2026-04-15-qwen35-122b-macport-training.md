# Qwen3.5-122B-A10B mac-port LoRA Training

**Date** : 2026-04-15
**Cible** : Qwen3.5-122B-A10B-BF16 sur Mac Studio M3 Ultra 512 GB
**Approche** : LoRA sur attention + linear_attn + shared_expert, BF16 full precision (pas de quantization)
**Statut** : Config validée par smoke — prête à lancer

## Contexte

Fine-tuner Qwen3.5-122B-A10B-BF16 pour distiller le reasoning Claude Opus, en top qualité sans OOM, sur Studio. Première mise au point de cette architecture hybride MoE + Mamba/SSM sur ce hardware.

## Architecture du modèle

- 48 hidden layers + 1 MTP = 49 transformer blocks
- 13 `self_attn` layers (full attention, q/k/v/o_proj)
- 36 `linear_attn` layers (Mamba-like SSM : in_proj_{a,b,qkv,z} + out_proj + conv1d + A_log + dt_bias)
- 256 MoE experts + 1 shared expert par layer, 8 experts routés par token
- Multimodal (vision + video tokens)

## Stratégie mac-port

Exploiter la mémoire unifiée Apple Silicon + paging macOS + mmap safetensors + Metal wired limit pour absorber les peaks de training sans OOM :

- `sudo sysctl -w iogpu.wired_limit_mb=458752` (448 GiB — plafonne Metal, laisse swap SSD pour le reste)
- safetensors lazy load (mmap à la demande)
- grad_checkpoint ON
- batch_size 1 + grad_accumulation_steps 8 (effective batch = 8)
- `clear_cache_threshold: 10` force `mx.metal.clear_cache()` toutes les 10 iters

## Config LoRA finale — 12 keys, rank 128 (Quality+)

```yaml
lora_parameters:
  rank: 128
  scale: 128.0     # alpha/rank = 1.0
  dropout: 0.01
  keys:
    # Full attention (13 layers)
    - self_attn.q_proj
    - self_attn.k_proj
    - self_attn.v_proj
    - self_attn.o_proj
    # Linear attention / Mamba SSM (36 layers)
    - linear_attn.in_proj_a
    - linear_attn.in_proj_b
    - linear_attn.in_proj_qkv
    - linear_attn.in_proj_z
    - linear_attn.out_proj
    # Shared expert MLP (48 layers, dense, voit 100 % des tokens)
    - mlp.shared_expert.gate_proj
    - mlp.shared_expert.up_proj
    - mlp.shared_expert.down_proj
```

**Rationale** :
- Full + linear attention = couverture de tous les 49 transformer blocks.
- Shared_expert est le MLP dense qui voit **100 % des tokens** (vs 3.1 % pour un expert MoE donné via routing 8/256) → gradient signal 32× plus fort. LoRA dessus = adaptation universelle, chef d'orchestre qui oriente les 256 experts frozen.
- Les 256 MoE experts sont volontairement exclus : les LoRA-iser fait matérialiser les experts en RAM Metal au fil des iters et crash Metal OOM au-delà de 463 GB (peak observé à iter 15 avec rank 64 sur gate/up/down_proj des experts).
- Trainable : ~337 M (0.276 % du modèle total).

## Pièges rencontrés

### Bug : `grad_accum_steps` ignoré

mlx-lm lit `grad_accumulation_steps` dans `TrainingArgs`, pas `grad_accum_steps`. La clé YAML `grad_accum_steps: 8` était silencieusement ignorée → effective batch = 1 au lieu de 8. À vérifier systématiquement dans le `adapter_config.json` après training.

### Crash Metal sur LoRA MoE experts

Première config avec 7 keys (attn + MLP gate/up/down) → Metal `Insufficient Memory` à iter 15 / peak 463 GB. Cause : matérialisation progressive des experts via routing (LoRA params + grads + Adam states + expert activations cumulés). 448 GiB wired-limit insuffisant. Fix : exclure les experts, n'utiliser que les projections attention + shared_expert.

### Kill silencieux jetsam

Deux smokes morts sans log d'erreur (iter 12 et iter 22). Signature jetsam macOS (kill -9 kernel-level sans trace applicative). Causes identifiées :
- iter 22 : peak memory jump (332 → 340 GB) + pression système
- iter 12 : un second training 122B concurrent (PID 18701) lancé pendant le smoke (deux 122B simultanés = 512 GB dépassé)

Garde-fous ajoutés : watchdog swap-thrash kill-switch, `clear_cache_threshold: 10`, `save_every: 100`.

### Script `train_offload.py` cassé

Script custom tentant l'offload SSD des 256 MoE experts. Deux bugs critiques :
1. Utilise la syntaxe JAX `output.at[].add()` → non supportée par MLX
2. `mlp.__call__ = ...` ne monkey-patche pas une instance Python (lookup dunder sur la classe)

Non fonctionnel, probablement jamais testé. Abandonné au profit du mac-port paging natif.

## Résultats des smokes

| Smoke | Keys | clear_cache | Iters atteints | Peak mem | Statut |
|-------|------|-------------|----------------|----------|--------|
| 1 | 7 (attn + MLP expert) | off | 15 | 463 GB | Metal OOM |
| 2 | 4 (attn only) | off | 1 | 252 GB | validé 1-iter |
| 3 | 9 (attn + linear_attn) | off | 22 | 340 GB | jetsam kill |
| 4 | 9 | 10 | 12 | 332 GB | kill par training concurrent |
| 5 | 9 | 10 | 32 | 340 GB | stable (kill manuel) |
| 6 (Q+) | 12 (attn + linear + shared) | 10 | 20 | 334 GB | nominal, 337 M trainable |

Dataset `final-opus-v3-1` : 12 506 samples (11 880 train + 626 valid), longueur mean 455 tokens, p50 281, p95 1207, p99 2036, max 7577. À seq 2048 : 99 % non tronqué.

## Layout des fichiers

```
configs/
  qwen35-122b-macport.yaml        # Config Quality+ committée
scripts/
  train_122b_macport.sh           # Wrapper : check wired-limit + watchdog + mlx_lm lora
  watchdog_mem.sh                 # Logger RSS/swap + kill-switch swap-thrash
tools/
  train_monitor_tui.py            # TUI rich : progression, loss, mem, rate, santé
logs/122b-macport/
  train-YYYYMMDD-HHMMSS.log       # Output mlx_lm
  watchdog-YYYYMMDD-HHMMSS.log    # Logs watchdog
  memcsv-YYYYMMDD-HHMMSS.csv      # Série temporelle mémoire
docs/plans/
  2026-04-15-qwen35-122b-macport-training.md  # Ce document
```

## Usage

### Prérequis (une seule fois)

```bash
sudo sysctl -w iogpu.wired_limit_mb=458752
# Persister au reboot :
echo 'iogpu.wired_limit_mb=458752' | sudo tee -a /etc/sysctl.conf
```

### Lancer le training

```bash
cd /Users/clems/ailiance-mac-tuner
./scripts/train_122b_macport.sh
# ou en détaché :
nohup ./scripts/train_122b_macport.sh > logs/122b-macport/nohup.out 2>&1 &
```

Le wrapper vérifie le wired-limit, lance le watchdog en arrière-plan, puis `mlx_lm lora --config configs/qwen35-122b-macport.yaml`.

### Monitorer en temps réel (TUI)

```bash
# Depuis Studio :
.venv/bin/python tools/train_monitor_tui.py --total-iters 5000

# Depuis une autre machine (TTY requis) :
ssh -t studio "cd ailiance-mac-tuner && .venv/bin/python tools/train_monitor_tui.py --total-iters 5000"
```

Affiche : progression iter/5000 + ETA, train loss sparkline + trajectory val, débit (it/s, tok/s, LR), mémoire (RSS, swap, peak Metal, free %), checks santé (loss NaN, peak proche limite, swap thrash), stream log récent.

### ETA observé

Avec `grad_accumulation_steps: 8` réellement appliqué, chaque iter = 8 micro-batches + 1 optimizer step, ~27 s/iter. Pour 5000 iters : **~25-37 h wall-clock**. Speed-up possible après iter 100 (compile cache stabilisé).

Pour accélérer au prix d'un peu de qualité :
- `iters: 3000` → ~20 h
- `grad_accumulation_steps: 4` → 50 % plus rapide, effective batch 4
- `rank: 64` → LoRA plus léger

### Post-training

```bash
# Fuse adapter → modèle complet BF16
.venv/bin/python -m mlx_lm fuse \
  --model models/Qwen3.5-122B-A10B-BF16 \
  --adapter-path output/qwen35-122b-macport \
  --save-path output/qwen35-122b-macport-fused --de-quantize

# Export GGUF Q4_K_M pour serving llama.cpp
cd ~/llama.cpp
python convert_hf_to_gguf.py ~/ailiance-mac-tuner/output/qwen35-122b-macport-fused \
  --outtype bf16 --outfile /tmp/qwen35-macport-BF16.gguf
./build/bin/llama-quantize /tmp/qwen35-macport-BF16.gguf \
  ~/ailiance-mac-tuner/output/qwen35-122b-macport-Q4_K_M.gguf Q4_K_M
```

## Troubleshooting

| Symptôme | Cause | Action |
|----------|-------|--------|
| `Metal: Insufficient Memory` | peak > wired_limit | augmenter `iogpu.wired_limit_mb` ou réduire rank/keys |
| Kill silencieux (pas de log) | jetsam | vérifier aucun training concurrent, lancer watchdog |
| Peak croît iter après iter | cache Metal non libéré | `clear_cache_threshold: 10` |
| Loss NaN à iter 1 | LR trop haut | `learning_rate: 4.0e-6`, warmup 200 |
| `KeyError` sur LoRA key | nom invalide | vérifier via `model.safetensors.index.json` |

## Commits de cette config

- `20131ac` — scaffold (wrapper, watchdog, TUI, YAML 4 keys)
- `5402597` — boost LoRA to 9 keys (linear_attn)
- `67a255e` — grad_accum fix + cache_clear + save100
- `61bdd03` — Quality+ rank128 + iters5000 + shared_expert

## Prochaines étapes

- [ ] Lancer le full 5000-iter run (fenêtre continue ~30 h)
- [ ] Post-training : fuse + GGUF Q4_K_M
- [ ] Éval qualité sur un benchmark reasoning (vs base, vs 35B-A3B)
