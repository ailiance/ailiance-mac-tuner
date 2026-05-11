# docs/

Documentation projet. Les specs et plans de travail datés vivent dans
`plans/` et `specs/`, une entrée par feature ou incident.

## Convention

- Un fichier par plan / décision, nommé `YYYY-MM-DD-<slug>.md`.
- Le slug décrit la feature (`ane-hybrid-pipeline`, `fix-oom-training`).
- Les plans contiennent : contexte, objectif, étapes, critères d'acceptation.

## Plans (`plans/`)

| Fichier | Objet |
|---|---|
| `2026-04-12-fix-oom-training.md` | Diagnostic + fix OOM training mlx-lm (wired-limit, mmap, grad-checkpoint, batch). |
| `2026-04-14-ane-hybrid-pipeline.md` | Pipeline hybride Qwen3.5-35B-A3B sur Apple Neural Engine (DeltaNet + Full Attention + MoE) — cf `research/ane-hybrid/`. |
| `2026-04-15-122b-opus-v3-training.md` | Training Qwen3.5-122B-A10B Opus-v3 (training final, val 0.468). |
| `2026-04-15-qwen35-122b-macport-training.md` | Profil mac-port LoRA rank 128, 5000 iters, 3.37 epochs. |
| `2026-04-15-mistral-small-opus-distill.md` | Distill Opus → Mistral Small. |
| `2026-04-15-devstral-v4-small-2-opus-distill.md` | Distill Opus → Devstral 2 24B (ailiance devstral-python). |
| `2026-04-15-micro-kiki-plan1-data-pipeline.md` | Plan 1 — pipeline data 1.57M → 63K (25 sources). |
| `2026-04-15-micro-kiki-plan2-brainstacks-training.md` | Plan 2 — Brainstacks 32 piles avec null-space projection. |
| `2026-04-15-micro-kiki-plan3-meta-router.md` | Plan 3 — meta-router 32 domaines (attention pooling, top-k=4). |
| `2026-04-15-micro-kiki-plan4-ane-pipeline.md` | Plan 4 — pipeline ANE pour micro-kiki. |

## Specs (`specs/`)

| Fichier | Objet |
|---|---|
| `2026-04-15-micro-kiki-design.md` | Architecture cible Brainstacks + Meta-router. |

## Recherche / SOTA

| Fichier | Objet |
|---|---|
| `sota-training-2026.md` | Techniques SOTA fine-tuning Apple Silicon Qwen3.5-122B. |
| `micro-kiki-moe-research.md` | Étude 32 LoRA experts déployables RTX 4090 24 GB. |
