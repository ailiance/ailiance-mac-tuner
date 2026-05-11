# Plan: Micro_KIKI — 32 Expert Brainstacks MoE

> **Date**: 2026-04-15
> **Author**: Clems (electron-rare) + Claude Opus 4.6
> **Status**: Draft — PRD v1.0
> **Hardware**: Mac M3 Ultra 512 Go + RTX 4090 24 Go (kxkm-ai)

---

## Context

### Problem Statement

The ailiance-mac-tuner project currently fine-tunes monolithic LLMs (Mistral Large 123B, Qwen3.5-122B) via LoRA on Apple Silicon. These models are excellent but impractical for deployment:

- **122B models** require 244 Go bf16 — only the Mac can run them
- **Single-LoRA approach** forces a jack-of-all-trades compromise across 32+ domains
- **No edge deployment** — kxkm-ai (RTX 4090, 24 Go) cannot serve 122B
- **No modularity** — adding a new domain requires retraining everything

### Solution: Micro_KIKI

A fleet of 32 specialized MoE-LoRA stacks on a frozen Qwen3.5-4B base, assembled via Brainstacks null-space projection and a sigmoid meta-router. Total inference footprint: ~4-5 Go VRAM.

### Why Now

1. **Qwen3.5-4B** (Feb 2026) delivers 79.1 MMLU-Pro — best-in-class 4B with GatedDeltaNet, 262K context, thinking mode, 201 languages
2. **Brainstacks** (April 2026) proves zero-forgetting continual learning with MoE-LoRA stacks and null-space projection
3. **Three teachers available**: Qwen3.5-122B Opus-v3 (val 0.468), Devstral 2 123B, Gemma 4 31B — all running on the Mac
4. **DeltaNet → CoreML** conversion proven at 474 tok/s/layer (ANE research Phase 1)
5. **Unsloth + mlx-tune** both support Qwen3.5 LoRA natively

### Constraints

| Constraint | Value |
|-----------|-------|
| Inference VRAM budget | < 8 Go (RTX 4090) |
| Base model size (Q4) | 2.5 Go |
| Max simultaneous stacks | 4 |
| Stack swap latency | < 2s |
| Router overhead | < 10 ms |
| Training hardware | Mac M3 Ultra 512 Go + RTX 4090 24 Go |
| Training time budget | < 80h total |
| Zero forgetting tolerance | Delta < 0.03 per domain |

---

## SOTA Research 2026 Integration

### Brainstacks (arxiv:2604.01152)

**Paper**: "Cross-Domain Cognitive Capabilities via Frozen MoE-LoRA Stacks for Continual LLM Learning" — April 2026

**Key findings for 32 experts**:

- Five interlocking components: MoE-LoRA blocks, inner residual boosting loop, outer continual curriculum loop, null-space projection via randomized SVD, outcome-based sigmoid meta-router
- Tested on Gemma 3 12B with 5 domains (chat, code, math, medical, reasoning) — we extend to 32 domains on Qwen3.5-4B
- Stacks encode "cognitive primitives" that transfer cross-domain — medical routing activates chat+math in 97% of cases
- Zero forgetting when domains evaluated in isolation

**Adaptations for Qwen3.5-4B (h_dim=3072, 32 domains)**:

| Parameter | Paper (Gemma 12B, 5 dom.) | Micro_KIKI (Qwen 4B, 32 dom.) | Rationale |
|-----------|--------------------------|-------------------------------|-----------|
| h_dim | 3840 | 3072 | Qwen3.5-4B architecture |
| ns_top_k_dirs | 64 | **32** | 32 domains x 32 dirs = 33% of 3072 space |
| MoE experts/stack | 4 | 4 | Proven configuration |
| LoRA rank | 16 | 16 | Balance capacity vs. VRAM |
| rsLoRA scaling | alpha=16, s=4.0 | Same | Paper default |
| Aux loss coeff | 0.01 | 0.01 | Paper default |
| Residual boost rounds | 2-3 | 1-2 | Smaller model saturates faster |
| Training steps/domain | 400-600 | ~500 | 2K examples, effective batch 16 |
| Stack size | 567 Mo (12B) | ~150 Mo (4B) | Linear scaling with h_dim |
| Total disk | 5.67 Go (5 dom.) | ~4.8 Go (32 dom.) | Smaller stacks, more domains |

**Null-space capacity analysis**:
- Available space: 3072 dimensions
- Per domain: 32 principal directions (ns_top_k_dirs=32)
- 32 domains x 32 dirs = 1024 directions consumed = **33% of null-space**
- Remaining capacity: 67% — comfortable margin for overlap and cross-domain transfer
- Risk threshold: >50% consumption may cause interference — monitor via forgetting check
- Mitigation: reduce to ns_top_k_dirs=24 if forgetting > 0.03 (uses only 25%)

**Null-space projection formula**:
```
P = V @ V.T          # V = top-K right singular vectors from rSVD
delta_proj = delta - delta @ P   # Remove frozen-stack directions
```
- nsamples = 400 validation examples from prior domains
- SVD method: `torch.svd_lowrank` when nsamples > 2K, else full SVD with truncation

### MoLoRA (arxiv:2603.15965)

**Paper**: "Composable Specialization via Per-Token Adapter Routing" — March 2026, Microsoft Research

**Comparison with Brainstacks for our use case**:

| Criterion | MoLoRA | Brainstacks | Verdict |
|-----------|--------|-------------|---------|
| Continual learning | Manual (retrain router) | **Native** (null-space) | Brainstacks |
| Zero forgetting guarantee | No formal guarantee | **Proven** via projection | Brainstacks |
| VRAM (32 experts Q4) | ~6-8 Go | ~4-5 Go (2-4 active) | Brainstacks |
| Routing granularity | Per-token | Per-prompt (sigmoid) | MoLoRA more granular |
| Multi-domain composition | top-K exclusive | **Multi-active sigmoid** | Brainstacks |
| Implementation maturity | PEFT integration | Paper only | MoLoRA |
| Training complexity | Joint all-experts | Sequential curriculum | MoLoRA simpler |

**When to use MoLoRA vs Brainstacks**:
- MoLoRA: when you need per-token routing and all experts fit in VRAM simultaneously
- Brainstacks: when you need continual learning, zero forgetting, and modular addition of domains
- **Decision: Brainstacks** — continual learning and zero forgetting are critical for a 32-domain system where we will iterate and add domains over months

**Hybrid approach**: Use MoLoRA-style per-token routing *within* each Brainstacks stack (the 4 experts per stack already use top-2 routing). The meta-router is per-prompt sigmoid (Brainstacks), the intra-stack routing is per-token top-2 (MoE-LoRA).

### DAPO/GRPO (arxiv:2503.14476, arxiv:2402.03300)

**GRPO** (Group Relative Policy Optimization — DeepSeek):
- Eliminates critic network entirely
- For each prompt: generate K responses (K=4-8), score with verifiable reward, normalize advantages by group mean/std
- Binary rewards: correct/incorrect for math, pass/fail for code tests, schema-valid for structured outputs
- Key insight: "RLVR begins to incentivize correct reasoning from the early stages of training"

**DAPO** (Decoupled Alignment via Process Optimization — ByteDance):
- Improves GRPO with clip-higher strategy to prevent entropy collapse
- Token-mean normalization for reward computation
- Trained Qwen-32B from ~0% to 50% on AIME 2024

**Verifiable rewards for coding experts** (applicable to stacks 2-14):

| Domain | Reward Function | Verification |
|--------|----------------|-------------|
| Python | Unit test pass rate | `subprocess.run(["python", "-c", code])` |
| TypeScript | Type check + test | `tsc --noEmit && npx jest` |
| C/C++ | Compile + run | `gcc -o test && ./test` |
| Rust | Cargo test | `cargo test` |
| SQL | Query validation | SQLite execute + result check |
| Shell | Syntax + dry-run | `bash -n script.sh` |
| KiCad DSL | S-expression parse | Custom validator |
| SPICE | Netlist syntax | ngspice parse check |

**Implementation on RTX 4090**:
- GRPO K=4 rollouts on Qwen3.5-4B QLoRA: ~14 Go VRAM — fits
- Use Unsloth GRPO implementation (native Qwen3.5 support)
- Batch of 8 prompts, K=4 each = 32 generations per step
- Estimated: ~2h per coding stack for GRPO phase

**On Mac M3 Ultra with ANE scorer pipeline**:
- GPU generates K=4 responses
- ANE scores each response in parallel via Qwen3.5-0.8B reward model
- Scoring overhead: ~0% (fully parallel on separate compute unit)
- Net throughput gain: scoring is "free"

### SimPO (arxiv:2405.14734)

**Key advantage for 4B model**: No reference model needed.

| Aspect | DPO | SimPO |
|--------|-----|-------|
| Reference model in memory | Required (2x VRAM) | **Not required** |
| VRAM for Qwen3.5-4B alignment | ~16 Go (2x 8 Go) | **~8 Go** |
| Reward signal | Log-ratio vs reference | Average log probability |
| Performance vs DPO | Baseline | **+6.4 pts AlpacaEval, +7.5 pts Arena-Hard** |

**Implementation details**:
- Hyperparameters: learning_rate, beta, gamma — total batch size 128 recommended
- Use after GRPO phase for general alignment of each stack
- Particularly important for non-verifiable domains (chat-fr, reasoning, general)
- mlx-tune 0.5+ has native SimPO support

**Per-stack SimPO training**:
- Generate 500 preference pairs per domain using teacher model
- Train for 1 epoch, lr=5e-7, beta=2.0, gamma=0.5
- ~30 min per stack on RTX 4090

### PiSSA (arxiv:2404.02948)

**SVD initialization for faster LoRA convergence**:
- Initializes LoRA A/B matrices from principal singular values/vectors of original weight matrix
- PiSSA outperforms standard LoRA by 5.16% on GSM8K (Mistral-7B)
- Fast SVD computation: seconds, not minutes
- NeurIPS 2024 Spotlight

**Applicable to MoE-LoRA stacks?**:
- Yes, with caveats. PiSSA can initialize each expert's LoRA within a stack
- Challenge: SVD must be computed per-projection (7 projections x 4 experts = 28 SVDs per stack)
- Benefit: faster convergence means fewer training steps (potentially 400 → 300 steps)
- Recommendation: **Use PiSSA init for Phase 2 stacks (coding core)** where convergence speed matters most, standard init for others
- PEFT library supports PiSSA via `init_lora_weights="pissa"` parameter

### DeltaNet to CoreML

**Our research results** (Phase 1 ANE research, 2026-04-14):
- Successfully converted Qwen3.5-9B DeltaNet layers to CoreML
- Throughput: **474 tok/s per layer** on ANE (M3 Ultra, 32 cores)
- Full model (40 layers): 14.4 tok/s on ANE alone
- MLX pure (full model): 14.2 tok/s — ANE matches MLX for single-model inference

**Application to 0.8B draft model**:
- Qwen3.5-0.8B uses same GatedDeltaNet architecture as 4B/9B
- Estimated CoreML size: ~1 Go
- Estimated ANE throughput: **200+ tok/s** (fewer layers, smaller hidden dim)
- Same tokenizer as 4B (same Qwen3.5 family) — critical for speculative decoding

**Speculative decoding potential**:
```
ANE (draft 0.8B):  propose [t1, t2, t3, t4, t5]  → 200+ tok/s
GPU (4B + stacks): verify  [t1, t2, t3, t4]       → 1 forward pass
Result:            accept 3-4 tokens per step       → 2-3x effective throughput
```

- Without speculative: 30-50 tok/s (GPU alone)
- With speculative ANE: **60-100 tok/s effective** (2-3x speedup)
- Zero additional VRAM (ANE has dedicated memory path)
- Zero additional power (~2W for ANE vs 20W for GPU)

**CoreML conversion path**:
1. Export Qwen3.5-0.8B PyTorch → ONNX (standard)
2. Convert ONNX → CoreML via coremltools (with DeltaNet custom ops)
3. Optimize: W8A8 quantization for ANE int8-int8 path (M3 Ultra supports this)
4. Alternatively: use ANEMLL framework (already installed at `/tmp/anemll`)

### Progressive Distillation

**Chain: 122B → 35B → 4B**:

| Hop | Teacher | Student | Quality Retention | Evidence |
|-----|---------|---------|-------------------|----------|
| 122B → 35B | Qwen3.5-122B Opus-v3 | Qwen3.5-35B-A3B | ~92-95% | Same MoE architecture, proven in literature |
| 35B → 4B | Qwen3.5-35B distilled | Qwen3.5-4B | ~85-90% | Same DeltaNet arch, thinking mode preserved |
| **Net 122B → 4B** | — | — | **~80-88%** | Progressive > direct (bridges capacity gap) |

**Why progressive beats direct**:
- Direct 122B → 4B: capacity gap too large, student cannot absorb teacher's logit distribution
- Intermediate 35B "bridges" the gap — its representations are closer to 4B's capacity
- Research confirms: multi-hop distillation retains 5-10% more quality than single-hop

**Multi-teacher ensemble benefits**:

| Domains | Primary Teacher | Secondary Teacher | Rationale |
|---------|----------------|-------------------|-----------|
| Coding (stacks 2-14) | Devstral 2 123B | Gemma 4 31B | Devstral: best code; Gemma: fast fallback |
| Embedded (stacks 15-25) | 122B Opus-v3 | Existing kiki-* data | Opus: domain expertise; kiki: real examples |
| Reasoning (stacks 1,16,31) | 122B Opus-v3 | Opus API (Claude) | Deep reasoning requires strongest teacher |
| Web/DevOps (stacks 26-30) | Gemma 4 31B | 122B Opus-v3 | Gemma: fast generation; Opus: quality check |
| Security (stack 32) | 122B Opus-v3 | — | Security requires highest accuracy |

**Per-domain distillation recipe**:
1. Generate 2K domain-specific prompts (manual + synthetic)
2. Run teacher to generate high-quality responses with `<think>` blocks
3. Filter: keep only responses where teacher confidence > 0.8
4. Deduplicate cross-domain (each example in exactly 1 domain)
5. Format as SFT dataset: `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`

---

## Objectives

### Must Have

| # | Objective | Metric | Target |
|---|-----------|--------|--------|
| M1 | 32 domain-specialized MoE-LoRA stacks trained and frozen | Stack count | 32 |
| M2 | Zero catastrophic forgetting across all domains | Max delta per domain | < 0.03 |
| M3 | Meta-router correctly routes to relevant stacks | F1 on routing eval | > 0.85 |
| M4 | Inference fits on RTX 4090 | Total VRAM | < 8 Go |
| M5 | Coding performance improvement over base | HumanEval pass@1 | > 70% (base: ~55%) |
| M6 | Reasoning improvement over base | GPQA-Diamond | > 80% (base: 76.2%) |
| M7 | Embedded domain accuracy | Custom eval (ESP-IDF, KiCad) | > 80% correct |
| M8 | French fluency | Custom eval (no code-switching) | Pass |
| M9 | Router latency | Wall-clock time | < 10 ms |
| M10 | Stack swap time | Hot-swap latency | < 2s |

### Must NOT

| # | Exclusion | Rationale |
|---|-----------|-----------|
| N1 | Do NOT use mergekit-moe for 32 experts | 80 Go VRAM — impossible on RTX 4090 |
| N2 | Do NOT train base model weights | Must remain frozen for stack composability |
| N3 | Do NOT use QLoRA 4-bit for Qwen3.5 training | Unsloth warns: "higher than normal quantization differences" |
| N4 | Do NOT skip null-space projection | Zero forgetting is a hard requirement |
| N5 | Do NOT use the same data across multiple stacks | Cross-domain dedup is mandatory |
| N6 | Do NOT hardcode router thresholds | Must be learned via outcome discovery |
| N7 | Do NOT deploy without forgetting regression test | Every stack addition must pass all prior evals |

### Nice to Have

| # | Feature | Benefit |
|---|---------|---------|
| H1 | ANE speculative decoding (0.8B draft) | 2-3x inference throughput on Mac |
| H2 | ANE GRPO scorer pipeline | Free scoring during RL training |
| H3 | ANE meta-router + embedding offload | Frees GPU for MoE compute only |
| H4 | PiSSA initialization for coding stacks | 20-25% faster convergence |
| H5 | SimPO alignment for all 32 stacks | Better preference alignment |
| H6 | Qwen3.5-9B fallback for complex tasks | Higher ceiling when 4B insufficient |
| H7 | vLLM serving with LoRA hot-swap | Production-ready inference |
| H8 | GGUF export for llama.cpp deployment | Broader compatibility |

---

## Architecture

### 4 Sub-MoE Design

The 32 stacks are organized into 4 logical sub-MoEs, each with its own curriculum ordering:

```
┌─────────────────────────────────────────────────────────────────┐
│                    MICRO_KIKI ARCHITECTURE                       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              META-ROUTEUR SIGMOID (32 outputs)            │   │
│  │  Input: 0.45 × mid_hidden + 0.55 × last_hidden           │   │
│  │  → Linear(3072, 512) → Global Attn → 32 × Cross-Attn     │   │
│  │  → MLP fusion (GELU, dropout 0.1) → 32 sigmoid + temp    │   │
│  │  Params: ~2M | Latency: ~5ms CPU, ~2ms ANE               │   │
│  └───────────────────────┬──────────────────────────────────┘   │
│                          │ activates stacks (threshold 0.12)     │
│                          ▼                                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │           BASE MODEL: Qwen3.5-4B (frozen, Q4)            │   │
│  │    GatedDeltaNet | 262K ctx | thinking natif | 2.5 Go     │   │
│  ├──────────────────────────────────────────────────────────┤   │
│  │                                                           │   │
│  │  SUB-MOE 1: CODING (14 stacks)                           │   │
│  │  ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┐            │   │
│  │  │py   │ts   │cpp  │rust │html │shell│sql  │            │   │
│  │  ├─────┼─────┼─────┼─────┼─────┼─────┼─────┤            │   │
│  │  │yaml │kicad│spice│dock │lua  │web-f│web-b│            │   │
│  │  └─────┴─────┴─────┴─────┴─────┴─────┴─────┘            │   │
│  │                                                           │   │
│  │  SUB-MOE 2: EMBEDDED (11 stacks)                          │   │
│  │  ┌─────┬─────┬─────┬─────┬─────┬─────┐                  │   │
│  │  │embed│stm32│iot  │freec│platf│power│                  │   │
│  │  ├─────┼─────┼─────┼─────┼─────┼─────┤                  │   │
│  │  │emc  │dsp  │spice│elec │kicad│     │                  │   │
│  │  └─────┴─────┴─────┴─────┴─────┴─────┘                  │   │
│  │                                                           │   │
│  │  SUB-MOE 3: REASONING (3 stacks)                          │   │
│  │  ┌─────┬─────┬─────┐                                     │   │
│  │  │reas │math │secur│                                     │   │
│  │  └─────┴─────┴─────┘                                     │   │
│  │                                                           │   │
│  │  SUB-MOE 4: GENERAL (4 stacks)                            │   │
│  │  ┌─────┬─────┬─────┬─────┐                               │   │
│  │  │chat │devop│llm  │music│                               │   │
│  │  └─────┴─────┴─────┴─────┘                               │   │
│  │                                                           │   │
│  │  Each stack: 4 MoE-LoRA experts, rank 16, top-2 routing  │   │
│  │  ~150 Mo/stack | 2-4 stacks active simultaneously         │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Stack Detail (per domain)

Each of the 32 stacks is a MoE-LoRA module applied to all 7 transformer projections:

```
Stack "python":
  ├── q_proj: MoE-LoRA(4 experts, rank=16, top-2)
  ├── k_proj: MoE-LoRA(4 experts, rank=16, top-2)
  ├── v_proj: MoE-LoRA(4 experts, rank=16, top-2)
  ├── o_proj: MoE-LoRA(4 experts, rank=16, top-2)
  ├── gate_proj: MoE-LoRA(4 experts, rank=16, top-2)
  ├── up_proj: MoE-LoRA(4 experts, rank=16, top-2)
  └── down_proj: MoE-LoRA(4 experts, rank=16, top-2)

  Total per expert: rank 16 × 2 matrices × 7 projections = 224 parameters per hidden dim
  Total per stack: 4 experts × ~37.5M = ~150M params → ~150 Mo disk (bf16)
  Active per inference: top-2 experts → ~75M active params
```

### ANE Triple Pipeline (Mac M3 Ultra only)

```
┌──────────────────────────────────────────────────────────────┐
│                    MAC M3 ULTRA 512 Go                         │
│                                                                │
│  ┌─────────────────┐  ┌────────────────────┐  ┌────────────┐ │
│  │   GPU METAL      │  │   ANE (32 cores)    │  │    CPU     │ │
│  │   76 cores       │  │   ~2W, 200+ tok/s   │  │  24 cores  │ │
│  │                  │  │                      │  │            │ │
│  │ Pipeline A:      │  │ Pipeline B:          │  │ Pipeline C:│ │
│  │ Base Qwen3.5-4B  │  │ Draft Qwen3.5-0.8B  │  │ Meta-router│ │
│  │ + 2-4 active     │  │ (speculative decode) │  │ sigmoid    │ │
│  │   MoE stacks     │  │                      │  │ (~5 ms)    │ │
│  │                  │  │ Reward scorer         │  │            │ │
│  │ Main generation  │  │ (GRPO phase)         │  │ Stack      │ │
│  │                  │  │                      │  │ offload    │ │
│  │                  │  │ Embedding + router    │  │ management │ │
│  │                  │  │ (inference phase)     │  │            │ │
│  └─────────────────┘  └────────────────────┘  └────────────┘ │
│         ↕ unified memory (zero-copy)  ↕                        │
└──────────────────────────────────────────────────────────────┘

Modes:
  A. Inference: GPU(4B+stacks) + ANE(0.8B draft) + CPU(router)
     → 60-100 tok/s effective (2-3x vs GPU alone)

  B. GRPO Training: GPU(generate K=4) + ANE(score responses) + CPU(router)
     → Scoring overhead: ~0% (fully parallel)

  C. Batch Scoring: ANE(score dataset) alone
     → 14 tok/s continuous, no GPU needed
```

### Hardware Mapping

| Component | Mac M3 Ultra | RTX 4090 (kxkm-ai) |
|-----------|-------------|---------------------|
| **Training** | |  |
| SFT LoRA (per stack) | mlx-tune, bf16, ~30 min | Unsloth QLoRA, ~20 min |
| GRPO (per stack) | GPU gen + ANE score, ~2h | Unsloth GRPO, ~2h |
| SimPO (per stack) | mlx-tune native, ~30 min | Unsloth, ~20 min |
| Distillation (teacher) | 122B bf16 on Mac only | Cannot fit 122B |
| **Inference** | |  |
| Base model | bf16, 8 Go | Q4, 2.5 Go |
| Active stacks (2-4) | bf16, ~1.2 Go | Q4/fp16, 0.6-1.2 Go |
| Meta-router | ANE or CPU, ~2-5 ms | CPU, ~5 ms |
| Speculative draft | ANE (0.8B), 200+ tok/s | N/A (no ANE) |
| KV cache (4K ctx) | ~1 Go | ~0.5 Go |
| **Total inference** | ~11 Go (bf16) | **~4-5 Go (Q4)** |
| **Headroom** | 501 Go free | **19 Go free** |

### VRAM Budget (RTX 4090 inference)

| Component | VRAM | Notes |
|-----------|------|-------|
| Qwen3.5-4B Q4 base | 2.5 Go | Frozen, always loaded |
| Meta-router | 0.01 Go | ~2M params |
| 2-4 active stacks | 0.6-1.2 Go | Hot-loaded on demand |
| KV cache (4K ctx) | 0.5 Go | GatedDeltaNet: constant KV |
| Framework overhead | 0.5 Go | vLLM or custom |
| **Total** | **4.1-4.7 Go** | |
| **Margin** | **19.3-19.9 Go** | Massive headroom |

---

## Implementation Steps

### Phase 1: Data Pipeline (Plan 1) — 2 weeks

**Goal**: Generate and curate 64K training examples (2K per domain).

| Step | Task | Time | Machine |
|------|------|------|---------|
| 1.1 | Define 32 domain taxonomies with example prompts | 4h | Local |
| 1.2 | Audit existing datasets (Opus 11.8K, kiki-* 5K, coding 18K) | 2h | Local |
| 1.3 | Assign existing data to domains (1 example → 1 domain only) | 4h | Local |
| 1.4 | Generate domain-specific prompts (manual + LLM-assisted) | 8h | Local |
| 1.5 | Distill coding domains via Devstral 2 123B | 24h | Mac |
| 1.6 | Distill embedded domains via 122B Opus-v3 | 16h | Mac |
| 1.7 | Distill reasoning/general via 122B Opus-v3 + Opus API | 8h | Mac |
| 1.8 | Cross-domain deduplication (cosine sim > 0.85 = duplicate) | 2h | Local |
| 1.9 | Quality filter: teacher confidence > 0.8, length > 100 tokens | 1h | Local |
| 1.10 | Format all datasets as SFT JSONL with `<think>` blocks | 2h | Local |

**Files to create/modify**:
- `scripts/prepare_micro_kiki_data.py` — Main data pipeline
- `scripts/distill_domain.py` — Per-domain distillation from teacher
- `scripts/dedup_cross_domain.py` — Cross-domain deduplication
- `configs/micro-kiki/domains.yaml` — 32 domain definitions
- `data/micro-kiki/{domain}/train.jsonl` — 32 training datasets
- `data/micro-kiki/{domain}/valid.jsonl` — 32 validation datasets

**Deliverable**: 32 curated datasets, 2K examples each, zero cross-domain duplicates.

### Phase 2: Brainstacks Training (Plan 2) — 2 weeks

**Goal**: Train 32 MoE-LoRA stacks with null-space projection, zero forgetting.

| Step | Task | Time | Machine |
|------|------|------|---------|
| 2.1 | Port Brainstacks code from Gemma to Qwen3.5 (same transformers API) | 8h | Local |
| 2.2 | Implement null-space projection with randomized SVD | 4h | Local |
| 2.3 | Implement MoE-LoRA module (4 experts, rank 16, top-2, 7 projections) | 8h | Local |
| 2.4 | Implement residual boosting inner loop | 4h | Local |
| 2.5 | Implement BestStackCallback (spike_threshold=0.1, patience=4) | 2h | Local |
| 2.6 | Train Phase 1 stacks: chat-fr, reasoning (foundations) | 2h | Mac+RTX |
| 2.7 | Train Phase 2 stacks: python, typescript, cpp, rust (coding core) | 4h | Mac+RTX |
| 2.8 | Train Phase 3 stacks: html-css through lua-upy (coding secondary) | 8h | Mac+RTX |
| 2.9 | Train Phase 4 stacks: embedded through kicad-pcb (technical) | 11h | Mac+RTX |
| 2.10 | Train Phase 5 stacks: web-frontend through llm-orch (applications) | 5h | Mac+RTX |
| 2.11 | Train Phase 6 stacks: math, security (complements) | 2h | Mac+RTX |
| 2.12 | Residual boost: round 2 for stacks with delta > 0.002 | 4-8h | Mac+RTX |
| 2.13 | Full forgetting regression test (all 32 domains) | 4h | Mac |

**Training hyperparameters per stack**:
```yaml
base_model: Qwen/Qwen3.5-4B
quantization: bf16 (Mac) / fp16 (RTX, NOT QLoRA 4-bit)
moe_experts: 4
moe_top_k: 2
lora_rank: 16
lora_alpha: 16  # rsLoRA scaling s=4.0
lora_targets: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
aux_loss_coeff: 0.01
learning_rate: 1.5e-4
lr_scheduler: cosine
warmup_steps: 50
max_steps: 500
effective_batch_size: 16
max_seq_length: 1024
ns_top_k_dirs: 32
ns_nsamples: 400
residual_boost_threshold: 0.002
residual_boost_max_rounds: 2
```

**Curriculum ordering** (sequential, each domain cannot degrade prior ones):
```
Phase 1 — Foundations
  1. chat-fr         → instruction-following + French
  2. reasoning       → meta-reasoning, thinking chains

Phase 2 — Coding Core
  3. python          → main coding language
  4. typescript      → web + types
  5. cpp             → systems + embedded
  6. rust            → safety + concurrency

Phase 3 — Coding Secondary
  7. html-css        → frontend markup
  8. shell           → scripts, DevOps
  9. sql             → queries, schemas
  10. yaml-json      → configs, schemas
  11. docker         → containers
  12. kicad-dsl      → netlists, footprints (KiCad S-expression)
  13. spice          → SPICE simulations
  14. lua-upy        → embedded scripting

Phase 4 — Technical Domains
  15. embedded       → ESP-IDF, firmware
  16. stm32          → STM32 HAL, CubeMX
  17. iot            → MQTT, BLE, protocols
  18. freecad        → mechanical CAD
  19. platformio     → build system
  20. power          → power supplies, regulators
  21. emc            → EMC/EMI, filtering
  22. dsp            → signal processing
  23. spice-sim      → circuit simulation
  24. electronics    → analog, RF, components
  25. kicad-pcb      → PCB routing, DRC

Phase 5 — Applications
  26. web-frontend   → React, Vite, patterns
  27. web-backend    → FastAPI, Hono, Express
  28. music-audio    → audio DSP, TTS, instruments
  29. devops         → Docker, Tailscale, CI/CD
  30. llm-orch       → RAG, agents, LLM routing

Phase 6 — Complements
  31. math           → math/physics reasoning
  32. security       → crypto, auth, OWASP
```

**Files to create/modify**:
- `scripts/micro_kiki/brainstacks.py` — Core Brainstacks training loop
- `scripts/micro_kiki/moe_lora.py` — MoE-LoRA module implementation
- `scripts/micro_kiki/null_space.py` — Null-space projection via rSVD
- `scripts/micro_kiki/residual_boost.py` — Inner-loop boosting
- `scripts/micro_kiki/callbacks.py` — BestStackCallback, forgetting check
- `scripts/micro_kiki/train_stack.py` — Single stack training entry point
- `scripts/micro_kiki/train_all.sh` — Full curriculum training orchestrator
- `configs/micro-kiki/training.yaml` — Global training config
- `configs/micro-kiki/curriculum.yaml` — Curriculum ordering + per-domain overrides

**Deliverable**: 32 frozen MoE-LoRA stacks in `output/micro-kiki/stacks/`, zero forgetting verified.

### Phase 3: Meta-Router (Plan 3) — 1 week

**Goal**: Train sigmoid meta-router via outcome discovery.

| Step | Task | Time | Machine |
|------|------|------|---------|
| 3.1 | Implement meta-router architecture (~2M params) | 4h | Local |
| 3.2 | Generate mixed evaluation dataset (500 prompts, all domains) | 2h | Local |
| 3.3 | Outcome discovery: for each prompt, test all 32 stacks individually | 16h | Mac |
| 3.4 | Greedy oracle: find optimal stack combinations per prompt | 2h | Local |
| 3.5 | Train router: BCE loss, 8 epochs, cosine LR | 2h | RTX |
| 3.6 | Evaluate: F1 on held-out routing test set | 1h | Local |
| 3.7 | Calibrate thresholds: chat floor 0.20, gate threshold 0.12 | 1h | Local |

**Outcome discovery procedure** (per prompt):
```python
for prompt in mixed_dataset:
    loss_base = forward(base_model, prompt)
    for i, stack in enumerate(stacks_32):
        loss_with_stack = forward(base_model + stack, prompt)
        delta[i] = loss_base - loss_with_stack

    # Greedy search: add stacks that reduce loss > 0.01
    active = set()
    while True:
        best = argmax(delta[j] for j not in active)
        if delta[best] < 0.01: break
        active.add(best)

    # Target: 80% discovered + 20% prior label
    target = 0.8 * discovered_vector + 0.2 * label_vector
```

**Router architecture detail**:
```python
class MicroKikiRouter(nn.Module):
    def __init__(self, h_dim=3072, hidden=512, n_domains=32):
        self.proj = nn.Linear(h_dim, hidden)
        self.global_query = nn.Parameter(torch.randn(1, hidden))
        self.global_attn = nn.MultiheadAttention(hidden, 4)
        self.domain_queries = nn.Parameter(torch.randn(n_domains, hidden))
        self.domain_attn = nn.MultiheadAttention(hidden, 4)
        self.fusion = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1)
        )
        self.temperature = nn.Parameter(torch.ones(n_domains))

    def forward(self, mid_hidden, last_hidden):
        h = 0.45 * mid_hidden + 0.55 * last_hidden  # (seq, h_dim)
        h = self.proj(h)                              # (seq, 512)
        ctx, _ = self.global_attn(self.global_query, h, h)  # (1, 512)
        per_domain, _ = self.domain_attn(self.domain_queries, h, h)  # (32, 512)
        logits = self.fusion(per_domain).squeeze(-1)   # (32,)
        return torch.sigmoid(logits / self.temperature) # (32,) independent
```

**Files to create/modify**:
- `scripts/micro_kiki/router.py` — Meta-router model
- `scripts/micro_kiki/outcome_discovery.py` — Greedy oracle search
- `scripts/micro_kiki/train_router.py` — Router training loop
- `configs/micro-kiki/router.yaml` — Router hyperparameters
- `output/micro-kiki/router.safetensors` — Trained router weights

**Deliverable**: Trained meta-router with F1 > 0.85, routing latency < 10 ms.

### Phase 4: ANE Pipeline (Plan 4) — 1 week (Nice to Have)

**Goal**: Deploy triple pipeline on Mac M3 Ultra for 2-3x inference speedup.

| Step | Task | Time | Machine |
|------|------|------|---------|
| 4.1 | Convert Qwen3.5-0.8B to CoreML (reuse DeltaNet conversion) | 4h | Mac |
| 4.2 | Benchmark ANE throughput for 0.8B (target: 200+ tok/s) | 2h | Mac |
| 4.3 | Implement speculative decoding loop (ANE draft → GPU verify) | 8h | Mac |
| 4.4 | Convert meta-router to CoreML (trivial, small MLP) | 1h | Mac |
| 4.5 | Convert embedding layer to CoreML | 1h | Mac |
| 4.6 | Implement reward scorer head (0.8B + Linear reward head) | 4h | Mac |
| 4.7 | Integration test: full triple pipeline end-to-end | 4h | Mac |
| 4.8 | Benchmark: tok/s with vs without speculative decoding | 2h | Mac |

**CoreML models to produce**:

| Model | CoreML Size | ANE Throughput | Conversion Tool |
|-------|------------|----------------|-----------------|
| Qwen3.5-0.8B (draft) | ~1 Go | 200+ tok/s | ANEMLL or coremltools |
| Meta-router | ~8 Mo | < 1 ms | coremltools (trivial) |
| Embedding layer | ~50 Mo | < 0.5 ms | coremltools (trivial) |
| Reward scorer (0.8B + head) | ~1 Go | 14 tok/s | ANEMLL |

**Files to create/modify**:
- `scripts/micro_kiki/convert_coreml.py` — CoreML conversion pipeline
- `scripts/micro_kiki/speculative_decode.py` — ANE speculative decoding
- `scripts/micro_kiki/ane_scorer.py` — ANE reward scorer for GRPO
- `scripts/micro_kiki/triple_pipeline.py` — Full ANE+GPU+CPU orchestrator
- `research/ane-hybrid/micro-kiki-coreml/` — CoreML model artifacts

**Deliverable**: 60-100 tok/s effective throughput (vs 30-50 without).

### Phase 5: Alignment (SimPO + GRPO) — 1.5 weeks

**Goal**: Align each stack via verifiable rewards (coding) and preference optimization (general).

| Step | Task | Time | Machine |
|------|------|------|---------|
| 5.1 | Implement verifiable reward functions for each coding domain | 8h | Local |
| 5.2 | GRPO on coding stacks (2-14): K=4 rollouts, binary rewards | 26h | RTX (or Mac+ANE) |
| 5.3 | Generate preference pairs for non-coding stacks (teacher-based) | 8h | Mac |
| 5.4 | SimPO on all 32 stacks: 500 pairs each, 1 epoch | 16h | RTX |
| 5.5 | Re-run forgetting regression test | 4h | Mac |
| 5.6 | Iterate: re-train stacks that degraded > 0.03 | 4h | Mac+RTX |

**GRPO configuration**:
```yaml
algorithm: grpo
K_rollouts: 4
batch_size: 8  # 8 prompts × 4 rollouts = 32 generations/step
max_new_tokens: 512
reward_type: verifiable  # binary: 0 or 1
advantage_normalization: group  # mean/std within K group
clip_range: 0.2
clip_higher: true  # DAPO improvement
token_mean_normalization: true  # DAPO improvement
entropy_bonus: 0.01
epochs_per_stack: 3
learning_rate: 5e-6
```

**SimPO configuration**:
```yaml
algorithm: simpo
preference_pairs: 500  # per domain
beta: 2.0
gamma: 0.5
learning_rate: 5e-7
batch_size: 4
epochs: 1
total_batch_size: 128  # via gradient accumulation
```

**Files to create/modify**:
- `scripts/micro_kiki/rewards/` — Directory of reward functions
- `scripts/micro_kiki/rewards/code_executor.py` — Sandboxed code execution
- `scripts/micro_kiki/rewards/schema_validator.py` — JSON/YAML validation
- `scripts/micro_kiki/rewards/spice_validator.py` — SPICE netlist validation
- `scripts/micro_kiki/train_grpo.py` — GRPO training loop
- `scripts/micro_kiki/train_simpo.py` — SimPO training loop
- `scripts/micro_kiki/generate_preferences.py` — Teacher-based preference generation
- `configs/micro-kiki/grpo.yaml` — GRPO hyperparameters
- `configs/micro-kiki/simpo.yaml` — SimPO hyperparameters

**Deliverable**: All 32 stacks aligned, coding stacks pass GRPO verifiable rewards, zero forgetting maintained.

### Phase 6: Export + Deploy — 0.5 weeks

**Goal**: Package for production inference on RTX 4090 and Mac.

| Step | Task | Time | Machine |
|------|------|------|---------|
| 6.1 | Export all stacks to safetensors format | 2h | Local |
| 6.2 | Export router to safetensors | 0.5h | Local |
| 6.3 | Quantize base model to Q4_K_M GGUF | 1h | Local |
| 6.4 | Write inference script (vLLM or custom) | 8h | Local |
| 6.5 | Write stack hot-swap logic | 4h | Local |
| 6.6 | End-to-end benchmark on RTX 4090 | 4h | RTX |
| 6.7 | End-to-end benchmark on Mac (with ANE pipeline) | 4h | Mac |
| 6.8 | Documentation and README | 4h | Local |

**Deployment layout**:
```
output/micro-kiki/
├── base/
│   ├── Qwen3.5-4B-Q4.gguf          # 2.5 Go (RTX)
│   └── Qwen3.5-4B-bf16/            # 8 Go (Mac)
├── stacks/
│   ├── 01-chat-fr.safetensors      # ~150 Mo each
│   ├── 02-python.safetensors
│   ├── ...
│   └── 32-security.safetensors
├── router/
│   └── router.safetensors           # ~8 Mo
├── coreml/                          # Mac only
│   ├── draft-0.8B.mlpackage         # ~1 Go
│   ├── router.mlpackage             # ~8 Mo
│   └── embedding.mlpackage          # ~50 Mo
├── configs/
│   ├── domains.yaml
│   ├── routing_thresholds.yaml
│   └── inference.yaml
└── scripts/
    ├── serve_rtx.py                 # RTX 4090 inference
    ├── serve_mac.py                 # Mac + ANE triple pipeline
    └── benchmark.py                 # Evaluation suite
```

**Files to create/modify**:
- `scripts/micro_kiki/export.py` — Export pipeline
- `scripts/micro_kiki/serve_rtx.py` — RTX inference server
- `scripts/micro_kiki/serve_mac.py` — Mac inference with ANE
- `scripts/micro_kiki/benchmark.py` — Full evaluation suite

**Deliverable**: Production-ready deployment on both machines.

---

## Files to Modify

| File | Action | Phase |
|------|--------|-------|
| `scripts/prepare_micro_kiki_data.py` | Create | 1 |
| `scripts/distill_domain.py` | Create | 1 |
| `scripts/dedup_cross_domain.py` | Create | 1 |
| `configs/micro-kiki/domains.yaml` | Create | 1 |
| `configs/micro-kiki/training.yaml` | Create | 2 |
| `configs/micro-kiki/curriculum.yaml` | Create | 2 |
| `scripts/micro_kiki/brainstacks.py` | Create | 2 |
| `scripts/micro_kiki/moe_lora.py` | Create | 2 |
| `scripts/micro_kiki/null_space.py` | Create | 2 |
| `scripts/micro_kiki/residual_boost.py` | Create | 2 |
| `scripts/micro_kiki/callbacks.py` | Create | 2 |
| `scripts/micro_kiki/train_stack.py` | Create | 2 |
| `scripts/micro_kiki/train_all.sh` | Create | 2 |
| `scripts/micro_kiki/router.py` | Create | 3 |
| `scripts/micro_kiki/outcome_discovery.py` | Create | 3 |
| `scripts/micro_kiki/train_router.py` | Create | 3 |
| `configs/micro-kiki/router.yaml` | Create | 3 |
| `scripts/micro_kiki/convert_coreml.py` | Create | 4 |
| `scripts/micro_kiki/speculative_decode.py` | Create | 4 |
| `scripts/micro_kiki/ane_scorer.py` | Create | 4 |
| `scripts/micro_kiki/triple_pipeline.py` | Create | 4 |
| `scripts/micro_kiki/rewards/code_executor.py` | Create | 5 |
| `scripts/micro_kiki/rewards/schema_validator.py` | Create | 5 |
| `scripts/micro_kiki/rewards/spice_validator.py` | Create | 5 |
| `scripts/micro_kiki/train_grpo.py` | Create | 5 |
| `scripts/micro_kiki/train_simpo.py` | Create | 5 |
| `scripts/micro_kiki/generate_preferences.py` | Create | 5 |
| `configs/micro-kiki/grpo.yaml` | Create | 5 |
| `configs/micro-kiki/simpo.yaml` | Create | 5 |
| `scripts/micro_kiki/export.py` | Create | 6 |
| `scripts/micro_kiki/serve_rtx.py` | Create | 6 |
| `scripts/micro_kiki/serve_mac.py` | Create | 6 |
| `scripts/micro_kiki/benchmark.py` | Create | 6 |

---

## Risk Matrix

| # | Risk | Probability | Impact | Mitigation |
|---|------|-------------|--------|------------|
| R1 | 32 domains saturate null-space (33% at ns_top_k_dirs=32) | Medium | High | Reduce to ns_top_k_dirs=24 (25%), monitor forgetting per-stack |
| R2 | Qwen3.5-4B too small for 32 specializations | Low | Critical | Fallback: Qwen3.5-9B (5.5 Go Q4, still fits RTX 4090) |
| R3 | Brainstacks not tested with Qwen3.5 architecture | Medium | High | Port is straightforward (same transformers API); validate on 2 stacks first |
| R4 | kxkm-ai inaccessible (Tailscale flaky) | Medium | Medium | Train 100% on Mac, deploy GGUF via NFS when kxkm-ai available |
| R5 | QLoRA 4-bit degrades Qwen3.5 quality | High | High | Already mitigated: use bf16 on Mac, fp16 on RTX (Unsloth warns against QLoRA for Qwen3.5) |
| R6 | Outcome discovery for router too slow (32 forwards per prompt) | Medium | Medium | Batch forwards, use Mac GPU parallelism; can subsample to 16 stacks if needed |
| R7 | Verifiable rewards too noisy for GRPO | Low | Medium | Start with Python (pytest-based) which is deterministic; iterate reward functions |
| R8 | CoreML conversion for DeltaNet fails on 0.8B | Low | Low | Not on critical path (Nice to Have); ANE pipeline is bonus |
| R9 | Teacher distillation produces low-quality data | Medium | High | Quality filter (confidence > 0.8), human spot-check per domain |
| R10 | Stack hot-swap latency > 2s | Low | Medium | Preload likely-next stacks; SSD on both machines is fast |
| R11 | MoE-Sieve insight: many experts are cold | Medium | Medium | Profile routing after training; prune cold experts to save VRAM |
| R12 | Domain overlap causes routing confusion | Medium | Medium | Strict cross-domain dedup; outcome discovery handles overlap naturally |

---

## Timeline

```
Week 1-2: Phase 1 — Data Pipeline
  ├── W1: Domain taxonomy, audit existing data, assign to domains
  ├── W1-2: Distillation via teachers (Mac runs 24/7)
  └── W2: Dedup, quality filter, format datasets

Week 3-4: Phase 2 — Brainstacks Training
  ├── W3: Port Brainstacks to Qwen3.5, implement core modules
  ├── W3-4: Train 32 stacks (Mac+RTX parallel)
  ├── W4: Residual boost rounds
  └── W4: Forgetting regression test

Week 5: Phase 3 — Meta-Router
  ├── Outcome discovery (16h on Mac)
  ├── Train router (2h on RTX)
  └── Evaluate and calibrate

Week 5-6: Phase 4 — ANE Pipeline (parallel with Phase 5)
  ├── CoreML conversions
  ├── Speculative decoding implementation
  └── Triple pipeline integration

Week 6-7: Phase 5 — Alignment
  ├── W6: GRPO on coding stacks (26h)
  ├── W6-7: SimPO on all stacks (16h)
  └── W7: Forgetting re-check

Week 7-8: Phase 6 — Export + Deploy
  ├── Export and quantize
  ├── Inference scripts
  ├── Benchmarks
  └── Documentation

Dependencies:
  Phase 1 → Phase 2 → Phase 3 → Phase 5 → Phase 6
                                  Phase 4 (parallel, independent)

Total: ~8 weeks (65-80h compute time, spread across 2 machines)
```

---

## Acceptance Criteria

### Phase 1: Data Pipeline
- [ ] 32 domain datasets created with >= 2K examples each
- [ ] Zero cross-domain duplicates (cosine sim < 0.85)
- [ ] Teacher confidence > 0.8 on all retained examples
- [ ] `<think>` blocks present in reasoning-intensive domains
- [ ] Datasets formatted as valid JSONL

### Phase 2: Brainstacks Training
- [ ] 32 MoE-LoRA stacks trained and frozen
- [ ] Each stack: 4 experts, rank 16, top-2 routing, 7 projections
- [ ] Null-space projection active for all stacks after stack 1
- [ ] Forgetting delta < 0.03 for every prior domain after each new stack
- [ ] Residual boost applied where delta > 0.002
- [ ] Total stack storage < 5 Go on disk

### Phase 3: Meta-Router
- [ ] Router F1 > 0.85 on routing evaluation set
- [ ] Router latency < 10 ms on CPU, < 5 ms on ANE
- [ ] Chat floor (0.20) always active
- [ ] Max 4 stacks activated simultaneously
- [ ] Router size < 10 Mo

### Phase 4: ANE Pipeline (Nice to Have)
- [ ] Qwen3.5-0.8B successfully converted to CoreML
- [ ] ANE throughput > 150 tok/s for draft model
- [ ] Speculative decoding achieves > 1.5x throughput improvement
- [ ] GRPO scorer runs on ANE without GPU interference

### Phase 5: Alignment
- [ ] HumanEval pass@1 > 70% (base: ~55%)
- [ ] GPQA-Diamond > 80% (base: 76.2%)
- [ ] Custom embedded eval > 80% correct
- [ ] French eval: fluent, no code-switching
- [ ] Forgetting delta < 0.03 after alignment

### Phase 6: Export + Deploy
- [ ] RTX 4090 inference: < 8 Go VRAM, < 2s stack swap
- [ ] Mac inference: triple pipeline operational (if Phase 4 done)
- [ ] End-to-end latency: first token < 500 ms
- [ ] Benchmark suite covers all 32 domains

### Overall
- [ ] Total compute time < 80h
- [ ] Total disk footprint < 15 Go (base + stacks + router + CoreML)
- [ ] Zero forgetting across all 32 domains (delta < 0.03)
- [ ] Production serving on RTX 4090 with < 8 Go VRAM

---

## SOTA Tips & Tricks 2026

### Qwen3.5-Specific

1. **Do NOT use QLoRA 4-bit** on Qwen3.5 models (MoE or dense). Unsloth explicitly warns about "higher than normal quantization differences." Use bf16 on Mac, fp16 on RTX.

2. **Chat template matters**: Wrong chat template is the #1 reason fine-tuned Qwen3.5 models behave strangely. Always use the official Qwen3.5 chat template from the tokenizer.

3. **Thinking mode**: Keep `<think>` blocks in training data. Mix 75% reasoning-style examples with 25% direct answers to preserve both capabilities.

4. **System prompt routing**: Qwen3.5 responds exceptionally well to system instructions defining a persona. Use "You are a Senior Python Developer" style system prompts — this helps the model's internal routing even for dense models. For our MoE stacks, the meta-router handles this, but include persona-style system prompts in training data.

5. **Unsloth speedup**: Unsloth makes Qwen3.5 train 1.5x faster with 50% less VRAM than FlashAttention2 setups. Use it on RTX 4090.

### Brainstacks-Specific

6. **Null-space budget**: At h_dim=3072 with ns_top_k_dirs=32 and 32 domains, you consume 33% of the space. This is well within safe bounds (paper uses 8.3% for 5 domains). Monitor forgetting; drop to 24 dirs if needed.

7. **Curriculum order matters**: Train foundational skills first (chat, reasoning), then coding, then domain-specific. The paper proves that later stacks can compose earlier ones (medical = chat + math in 97% of cases).

8. **Residual boost is cheap**: Each round is ~10 min per stack. Always run at least 1 boost round — the paper shows consistent improvement of 0.003-0.01 loss.

9. **Router checkpoint selection**: Use composite score: 0.50 x accuracy + 0.35 x set_match - 0.15 x val_bce. This balances routing precision with routing recall.

10. **Reasoning soft-boost**: Set reasoning domain target to 0.5 (not 1.0) in the router training. Reasoning is a meta-skill that should partially activate alongside other domains, not exclusively.

### MoE-LoRA Routing

11. **MoE-Sieve insight**: After training, profile routing per layer. Top-25% most-routed experts per layer handle most tokens. Consider pruning cold experts (70-73% parameter reduction with <1% accuracy loss).

12. **Routing collapse prevention**: If using GRPO on MoE-LoRA, consider RO-GRPO (Routing-Optimized GRPO) which transforms routing statistics into reward signals to prevent expert underutilization.

13. **Layer-aware allocation** (from Lamer-SSL): Allocate more experts to deeper layers where representations are more specialized. For our fixed 4-expert-per-stack design, this translates to: focus LoRA rank budget on deeper layers if capacity becomes an issue.

### GRPO/DAPO Implementation

14. **GRPO K value**: K=4 is the sweet spot for 4B models on 24 Go VRAM. K=8 is better quality but requires ~22 Go — too tight.

15. **DAPO improvements over GRPO**: Always use clip-higher and token-mean normalization. These are simple additions that prevent entropy collapse, especially important during early training.

16. **Verifiable rewards are king**: For coding domains, binary pass/fail from test execution is more effective than any learned reward model. Start with Python (pytest) where verification is most reliable.

17. **RLVR insight (2026)**: "RLVR begins to incentivize correct reasoning from the early stages of training, and this capability generalizes well to unseen test questions." Start GRPO early — even 1 epoch helps.

### PiSSA Initialization

18. **Use PiSSA for coding stacks**: Initialize LoRA A/B matrices from principal SVD of original weights. 5%+ improvement on reasoning tasks, faster convergence (400 → 300 steps).

19. **Fast SVD is essential**: Use `torch.svd_lowrank` (not full SVD) for PiSSA initialization. Full SVD on 3072-dim matrices is 10x slower with negligible quality difference.

### Distillation

20. **Progressive beats direct**: 122B → 35B → 4B retains 80-88% quality. Direct 122B → 4B retains only ~70-75%. The 35B intermediate is critical.

21. **Multi-teacher ensemble**: Use different teachers for different domains. Devstral 2 123B for code, Opus-v3 for reasoning/embedded, Gemma 4 31B for fast web/DevOps generation. The diversity improves student robustness.

22. **Include 10-20% general data in every domain**: Prevents expert over-specialization. Even the most niche stack (e.g., emc) should see some general instruction-following data.

### ANE / Apple Silicon

23. **W8A8 on ANE**: For CoreML models on M3 Ultra, use int8 weights + int8 activations. This leverages the faster int8-int8 compute path, giving better latency than fp16 on ANE.

24. **M5 Neural Accelerators**: The M5 has GPU-integrated neural accelerators that are 3.3-4x faster for time-to-first-token vs M4. If upgrading, the speculative decoding pipeline becomes even more effective.

25. **Speculative decoding acceptance rate**: With same-family models (0.8B draft, 4B target), expect 60-75% token acceptance rate. This translates to 2-3x effective throughput.

### General

26. **One example, one domain**: Strict cross-domain deduplication is critical. The null-space projection handles knowledge transfer; duplicated training data would waste capacity.

27. **Batch size 16 effective**: Use gradient accumulation. Physical batch = 2-4 on RTX, accumulate to 16. Larger batches (32+) showed diminishing returns in Brainstacks paper.

28. **Cosine LR with warmup**: 50 steps warmup, cosine decay to 0. Standard but important — linear decay loses 1-2% quality vs cosine.

29. **Forgetting check after every stack**: The forgetting regression test takes ~7 min (32 quick evals). Run it after every new stack, not just at the end. Catch regressions early.

30. **Save checkpoints aggressively**: Each stack is ~150 Mo. Save after every stack training + after every residual boost round. Disk is cheap (total ~5 Go); retraining is expensive (~30 min/stack).

---

## Sources

- [Brainstacks: Cross-Domain Cognitive Capabilities via Frozen MoE-LoRA Stacks](https://arxiv.org/html/2604.01152)
- [MoLoRA: Composable Specialization via Per-Token Adapter Routing](https://arxiv.org/abs/2603.15965)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/pdf/2503.14476)
- [GRPO: Group Relative Policy Optimization](https://arxiv.org/abs/2402.03300)
- [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734)
- [PiSSA: Principal Singular Values and Singular Vectors Adaptation](https://arxiv.org/abs/2404.02948)
- [MoE-Sieve: Routing-Guided LoRA for Efficient MoE Fine-Tuning](https://arxiv.org/abs/2603.24044)
- [Balancing the Experts: LoRA-MoE for GRPO via Mechanism-Aware Rewards](https://openreview.net/forum?id=rhD7ZuFAjU)
- [Lamer-SSL: Layer-aware Mixture of LoRA Experts](https://arxiv.org/html/2602.12746)
- [GOAT: Bridging LoRA vs Full Fine-Tuning with Adaptive MoE and SVD](https://medium.com/@lakhanmalviya_91647/goat-bridging-the-lora-vs-full-fine-tuning-gap-with-adaptive-moe-and-svd-0b2635144194)
- [Post-Training in 2026: GRPO, DAPO, RLVR & Beyond](https://llm-stats.com/blog/research/post-training-techniques-2026)
- [Qwen3.5 Fine-Tuning Guide — Unsloth](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
- [Apple Neural Engine for LLM Inference](https://insiderllm.com/guides/apple-neural-engine-llm-inference/)
- [Inside the M4 Apple Neural Engine Benchmarks](https://maderix.substack.com/p/inside-the-m4-apple-neural-engine-615)
- [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers)
- [Dynamic Expert Specialization for Multi-Domain MoE](https://arxiv.org/html/2509.16882)
- [TT-LoRA MoE: Parameter-Efficient Fine-Tuning and Sparse MoE](https://dl.acm.org/doi/10.1145/3712285.3759888)
- [Progressive Distillation — Emergent Mind](https://www.emergentmind.com/topics/progressive-distillation)
