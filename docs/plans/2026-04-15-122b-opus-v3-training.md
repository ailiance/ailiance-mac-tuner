# Qwen3.5-122B-A10B-Opus-v3 Training Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tuner Qwen3.5-122B-A10B BF16 avec LoRA rank 64 sur M3 Ultra 512 Go pour creer le premier modele Opus-v3 122B.

**Architecture:** Trois pistes sequentielles : (1) mlx-tune MoE-aware training, (2) recompiler MLX pour augmenter la limite Metal 499K, (3) fallback seq_length 896. Piste 1 testee en premier car c'est le path of least resistance.

**Tech Stack:** mlx-tune 0.4.21, MLX, Apple Silicon M3 Ultra 512 Go, Qwen3.5-122B-A10B BF16

---

## File Structure

| Fichier | Responsabilite |
|---------|----------------|
| `scripts/train_122b_mlxtune.py` | Training via mlx-tune (Piste 1) |
| `configs/mlx-lm-qwen35-122b-opus-v3.yaml` | Config mlx_lm (Pistes 2-3) |
| `data/final-opus-v3-1/train.jsonl` | 11880 exemples training |
| `data/final-opus-v3-1/valid.jsonl` | 626 exemples validation |
| `output/qwen35-122b-opus-v3/` | Adapters LoRA sortie |
| `models/Qwen3.5-122B-A10B-BF16/` | Modele base (233 Go, 39 shards) |

---

## Piste 1 : mlx-tune (recommandee)

### Task 1: Script training mlx-tune

**Files:**
- Create: `scripts/train_122b_mlxtune.py`

- [ ] **Step 1: Ecrire le script**

```python
#!/usr/bin/env python3
"""Training 122B Opus-v3 via mlx-tune (MoE-aware LoRA)."""

import os
os.environ['PYTHONUNBUFFERED'] = '1'

from mlx_tune import FastLanguageModel, SFTConfig, SFTTrainer
from datasets import load_dataset

# 1. Charger le modele avec LoRA
print("Chargement Qwen3.5-122B-A10B-BF16...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="models/Qwen3.5-122B-A10B-BF16",
    max_seq_length=1280,
    dtype=None,  # auto-detect bf16
    load_in_4bit=False,  # bf16 complet
)

# 2. Appliquer LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=64,
    lora_alpha=128,
    lora_dropout=0.01,
    target_modules=[
        "in_proj_qkv", "in_proj_z", "in_proj_a",
        "in_proj_b", "out_proj",  # DeltaNet layers
        "q_proj", "k_proj", "v_proj", "o_proj",  # Full attention
    ],
    use_gradient_checkpointing="unsloth",
)

# 3. Dataset
dataset = load_dataset(
    "json",
    data_files={
        "train": "data/final-opus-v3-1/train.jsonl",
        "validation": "data/final-opus-v3-1/valid.jsonl",
    },
)

def format_example(example):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}

dataset = dataset.map(format_example)

# 4. Training config
config = SFTConfig(
    output_dir="output/qwen35-122b-opus-v3",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=8e-6,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    num_train_epochs=2,
    max_seq_length=1280,
    logging_steps=5,
    save_steps=50,
    save_total_limit=3,
    optim="adamw_8bit",
    dataset_text_field="text",
    packing=False,
)

# 5. Trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    args=config,
)

# 6. Train
print("Lancement training...")
trainer.train()

# 7. Save
model.save_pretrained("output/qwen35-122b-opus-v3/final")
tokenizer.save_pretrained("output/qwen35-122b-opus-v3/final")
print("Training termine.")
```

- [ ] **Step 2: Installer datasets (dependance)**

Run: `source .venv/bin/activate && uv pip install datasets`
Expected: Installation OK

- [ ] **Step 3: Tester le chargement (5 min max)**

Run: `source .venv/bin/activate && timeout 300 python scripts/train_122b_mlxtune.py 2>&1 | head -30`

Expected: Le modele charge, LoRA applique, training demarre.
Si OOM ou crash Metal → passer a Piste 2 (Task 3).

- [ ] **Step 4: Si OK, lancer le training complet**

Run:
```bash
source .venv/bin/activate
nohup python scripts/train_122b_mlxtune.py > logs/mlxtune-122b.log 2>&1 &
echo "PID: $!" && echo "TUI: python scripts/training_tui.py logs/mlxtune-122b.log"
```

- [ ] **Step 5: Commit**

```bash
git add scripts/train_122b_mlxtune.py
git commit -m "feat: mlx-tune training script for 122B Opus v3"
```

---

## Piste 2 : Recompiler MLX (limite Metal 499K)

Si Piste 1 crashe avec `Resource limit (499000) exceeded`.

### Task 2: Identifier la limite exacte dans MLX

**Files:**
- Modify: `/tmp/mlx-fork/mlx/backend/metal/allocator.cpp`

- [ ] **Step 1: Cloner MLX**

Run:
```bash
cd /tmp
git clone --depth 1 https://github.com/ml-explore/mlx.git mlx-fork
```

- [ ] **Step 2: Trouver le code de la limite**

Run:
```bash
grep -rn "block_limit\|499000\|recommendedMaxWorkingSetSize\|resource_limit" /tmp/mlx-fork/mlx/backend/metal/
```

Expected: Fichier `allocator.cpp` avec la ligne de la limite.

- [ ] **Step 3: Doubler la limite**

Le fix depend de ce qu'on trouve a l'etape 2. Pattern probable:
```cpp
// Avant:
size_t block_limit = device_->recommendedMaxWorkingSetSize();
// Apres:
size_t block_limit = 2 * device_->recommendedMaxWorkingSetSize();
```

Ou si c'est un hardcode:
```cpp
// Avant:
constexpr size_t MAX_BUFFERS = 499000;
// Apres:
constexpr size_t MAX_BUFFERS = 998000;
```

- [ ] **Step 4: Compiler et installer**

Run:
```bash
cd /tmp/mlx-fork
pip install -e ".[metal]"
```

Expected: Compilation ~5 min, installation OK.

- [ ] **Step 5: Tester avec le training**

Run:
```bash
source /Users/clems/ailiance-mac-tuner/.venv/bin/activate
python -m mlx_lm lora --config configs/mlx-lm-qwen35-122b-opus-v3.yaml 2>&1 | head -30
```

Expected: Depasse iter 20 sans crash Metal.
Si toujours OOM memoire (pas handles) → passer a Piste 3.

- [ ] **Step 6: Commit**

```bash
cd /Users/clems/ailiance-mac-tuner
git add configs/mlx-lm-qwen35-122b-opus-v3.yaml
git commit -m "fix: recompiled MLX with 2x Metal buffer limit"
```

---

## Piste 3 : Fallback seq_length 896

Si les deux pistes precedentes echouent.

### Task 3: Reduire la sequence length

**Files:**
- Modify: `configs/mlx-lm-qwen35-122b-opus-v3.yaml`

- [ ] **Step 1: Modifier la config**

```yaml
# Changer max_seq_length de 1280 a 896
max_seq_length: 896
# Aussi reduire grad_accumulation pour compenser
grad_accumulation_steps: 16
```

- [ ] **Step 2: Filtrer le dataset (exemples trop longs)**

Run:
```bash
source .venv/bin/activate
python -c "
import json
from pathlib import Path
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('models/Qwen3.5-122B-A10B-BF16')
data = Path('data/final-opus-v3-1')

for split in ['train', 'valid']:
    src = data / f'{split}.jsonl'
    dst = data / f'{split}_short.jsonl'
    kept = 0
    total = 0
    with open(src) as fin, open(dst, 'w') as fout:
        for line in fin:
            total += 1
            d = json.loads(line)
            msgs = d['messages']
            text = tokenizer.apply_chat_template(msgs, tokenize=False)
            tokens = tokenizer.encode(text)
            if len(tokens) <= 896:
                fout.write(line)
                kept += 1
    print(f'{split}: {kept}/{total} exemples gardes (<896 tokens)')
"
```

- [ ] **Step 3: Mettre a jour la config pour le dataset filtre**

Si necessaire, pointer vers les fichiers filtres ou garder le dataset
original (mlx_lm tronque automatiquement).

- [ ] **Step 4: Lancer le training**

Run:
```bash
source .venv/bin/activate
python -m mlx_lm lora --config configs/mlx-lm-qwen35-122b-opus-v3.yaml 2>&1 | tee logs/pipeline-122b.log
```

Expected: Training stable, depasse iter 100.
Peak memory: ~350-400 Go (vs 463 Go avec seq 2048).

- [ ] **Step 5: Commit**

```bash
git add configs/mlx-lm-qwen35-122b-opus-v3.yaml
git commit -m "fix: reduce seq_length to 896 for 122B BF16 training"
```

---

## Ordre d'execution

```
Piste 1 (mlx-tune)
    |
    +-- OK → Training complet → FIN
    |
    +-- FAIL (Metal 499K) → Piste 2 (recompiler MLX)
                                |
                                +-- OK → Training complet → FIN
                                |
                                +-- FAIL (OOM RAM) → Piste 3 (seq 896)
                                                        |
                                                        +-- OK → Training complet → FIN
```

## Criteres de succes

- [ ] Training depasse iter 100 sans crash
- [ ] Val loss < 0.8 a iter 50
- [ ] Peak memory < 490 Go
- [ ] Adapters sauvegardes dans `output/qwen35-122b-opus-v3/`
