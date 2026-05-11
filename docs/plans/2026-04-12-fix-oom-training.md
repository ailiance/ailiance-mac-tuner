# Fix Metal OOM During Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the Metal Out-of-Memory crash that occurs at first validation pass during Mistral Large 123B LoRA fine-tuning.

**Architecture:** Two root causes — deprecated Metal memory API (`mx.metal.set_memory_limit`) that may not enforce limits correctly, and hardcoded `val_batches=25` causing excessive memory during validation. Fix both in `scripts/train_mlx.py`, add `val_batches` to config YAML, and verify the venv has MLX installed.

**Tech Stack:** Python 3.12, MLX >=0.31.0, mlx-lm >=0.31.0, Apple Silicon M4 Pro 512 Go

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `scripts/train_mlx.py` | Modify lines 221-222, 289 | Fix memory API + val_batches |
| `configs/mistral-large.yaml` | Modify | Add `val_batches: 5` |
| `configs/mistral-small.yaml` | Modify | Add `val_batches: 10` |
| `configs/qwen-27b.yaml` | Modify | Add `val_batches: 10` |

---

### Task 1: Verify and fix venv

**Files:**
- Check: `.venv/bin/python`, `requirements.txt`

- [ ] **Step 1: Check if MLX is installed**

```bash
cd /Users/clems/ailiance-mac-tuner
source .venv/bin/activate
python3 -c "import mlx; print(mlx.__version__)"
```

Expected: Either prints version >=0.31.0, or ImportError.

- [ ] **Step 2: Reinstall deps if missing**

```bash
cd /Users/clems/ailiance-mac-tuner
source .venv/bin/activate
uv pip install -r requirements.txt
```

Expected: All packages installed, no errors.

- [ ] **Step 3: Verify MLX version and new API availability**

```bash
source .venv/bin/activate
python3 -c "import mlx.core as mx; print(mx.__version__); print(hasattr(mx, 'set_memory_limit'))"
```

Expected: Version >=0.31.0, `True` for `set_memory_limit`.

- [ ] **Step 4: Commit if requirements changed**

No code changed — skip commit if only venv was rebuilt.

---

### Task 2: Fix deprecated Metal memory API

**Files:**
- Modify: `scripts/train_mlx.py:219-225`

- [ ] **Step 1: Locate current code**

Lines 219-225 currently read:

```python
        # Cap Metal memory usage (leave headroom for OS)
        mem_limit_gb = config.get("memory_limit_gb", 460)
        mx.metal.set_memory_limit(mem_limit_gb * 1024**3)
        mx.metal.set_cache_limit(32 * 1024**3)
        print(f"Metal memory limit: {mem_limit_gb} GB, cache limit: 32 GB")
    except Exception:
        pass
```

- [ ] **Step 2: Replace with new API**

Replace lines 221-222 with:

```python
        mx.set_memory_limit(mem_limit_gb * 1024**3)
        mx.set_cache_limit(32 * 1024**3)
```

The rest of the block stays identical. `mx.set_memory_limit()` is the non-deprecated API available in MLX >=0.31.0.

- [ ] **Step 3: Verify no other deprecated calls exist**

```bash
grep -n "mx\.metal\." scripts/train_mlx.py
```

Expected: No matches.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_mlx.py
git commit -m "fix: use non-deprecated mx.set_memory_limit API

mx.metal.set_memory_limit() is deprecated in MLX 0.31+ and may not
enforce limits correctly, contributing to Metal OOM during validation."
```

---

### Task 3: Make val_batches configurable and reduce default

**Files:**
- Modify: `scripts/train_mlx.py:286-297`

- [ ] **Step 1: Locate current TrainingArgs**

Lines 286-297 currently read:

```python
    train_args = TrainingArgs(
        batch_size=batch_size,
        iters=total_iters,
        val_batches=25,
        steps_per_report=5,
        steps_per_eval=config.get("save_every", 50),
        steps_per_save=config.get("save_every", 50),
        max_seq_length=config.get("max_seq_length", 4096),
        adapter_file=str(output_dir / "adapters.safetensors"),
        grad_checkpoint=True,
        grad_accumulation_steps=grad_accum,
    )
```

- [ ] **Step 2: Replace hardcoded val_batches with config lookup**

Change line 289 from:

```python
        val_batches=25,
```

to:

```python
        val_batches=config.get("val_batches", 5),
```

Default is 5 (not 25). For 123B models, 5 validation batches is sufficient to estimate val_loss without exhausting Metal memory. Configs can override.

- [ ] **Step 3: Verify the edit**

```bash
grep -n "val_batches" scripts/train_mlx.py
```

Expected: One match at line ~289 showing `config.get("val_batches", 5)`.

- [ ] **Step 4: Commit**

```bash
git add scripts/train_mlx.py
git commit -m "fix: make val_batches configurable, reduce default from 25 to 5

val_batches=25 was hardcoded, causing Metal OOM on Mistral Large 123B
during first validation pass. Now reads from config YAML with safe
default of 5 batches."
```

---

### Task 4: Add val_batches to config YAMLs

**Files:**
- Modify: `configs/mistral-large.yaml`
- Modify: `configs/mistral-small.yaml`
- Modify: `configs/qwen-27b.yaml`

- [ ] **Step 1: Add val_batches to mistral-large.yaml**

After `memory_limit_gb: 400`, add:

```yaml
val_batches: 5
```

5 is conservative for 123B. Enough to estimate val_loss trend for early stopping.

- [ ] **Step 2: Add val_batches to mistral-small.yaml**

Add under the training section:

```yaml
val_batches: 10
```

24B models have more memory headroom — 10 is safe.

- [ ] **Step 3: Add val_batches to qwen-27b.yaml**

Add under the training section:

```yaml
val_batches: 10
```

Same reasoning as Mistral Small.

- [ ] **Step 4: Commit**

```bash
git add configs/mistral-large.yaml configs/mistral-small.yaml configs/qwen-27b.yaml
git commit -m "feat: add val_batches to all model configs

mistral-large: 5 (conservative for 123B OOM prevention)
mistral-small, qwen-27b: 10 (more headroom at 24-27B)"
```

---

### Task 5: Smoke test — dry run validation

**Files:**
- None modified, verification only

- [ ] **Step 1: Activate venv and run syntax check**

```bash
cd /Users/clems/ailiance-mac-tuner
source .venv/bin/activate
python3 -c "import scripts.train_mlx" 2>&1 || python3 -c "exec(open('scripts/train_mlx.py').read().split('if __name__')[0])"
```

Expected: No syntax errors, no import errors.

- [ ] **Step 2: Verify config loads correctly**

```bash
source .venv/bin/activate
python3 -c "
import yaml
with open('configs/mistral-large.yaml') as f:
    c = yaml.safe_load(f)
assert c['val_batches'] == 5
assert c['memory_limit_gb'] == 400
assert c['lora_rank'] == 48
print('Config OK:', {k: c[k] for k in ['val_batches', 'memory_limit_gb', 'lora_rank', 'batch_size']})
"
```

Expected: `Config OK: {'val_batches': 5, 'memory_limit_gb': 400, 'lora_rank': 48, 'batch_size': 1}`

- [ ] **Step 3: Launch training**

```bash
cd /Users/clems/ailiance-mac-tuner
./train.sh 2>&1 | tee training.log
```

Watch for:
- "Metal memory limit: 400 GB" (confirms new API works)
- No deprecation warning about `mx.metal.set_memory_limit`
- First validation pass completes (5 batches) without OOM
- Training continues to step 10+

- [ ] **Step 4: If OOM persists — further reduce**

If still OOM at validation, reduce `val_batches: 2` in config and retry.
If OOM during training (not validation), the issue is elsewhere — check `lora_rank` and `batch_size`.

---

## Summary of Changes

| What | Before | After |
|------|--------|-------|
| Memory API | `mx.metal.set_memory_limit()` (deprecated) | `mx.set_memory_limit()` |
| Cache API | `mx.metal.set_cache_limit()` (deprecated) | `mx.set_cache_limit()` |
| val_batches | Hardcoded 25 | Config-driven, default 5 |
| mistral-large val_batches | N/A | 5 |
| mistral-small val_batches | N/A | 10 |
| qwen-27b val_batches | N/A | 10 |
