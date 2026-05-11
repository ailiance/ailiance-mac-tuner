# tools/

Outils autonomes (monitoring, TUI, utilitaires) utilisables hors pipeline.

## Fichiers

| Fichier | Role |
|---|---|
| `train_monitor_tui.py` | TUI live (rich) pour le training Qwen3.5-122B-A10B-BF16 mac-port. Tail le log `mlx_lm.lora` + `memcsv` watchdog, affiche progress bar, iter/ETA, sparkline loss, trajectoire val, memoire (RSS/swap/peak Metal), rate (it/s, tok/s, LR), health checks et log recent. |
| `archive_dead_artifacts.sh` | Archive vers `_archive/<date>/` les adapters morts/orphelins identifiés à l'audit 2026-05-04 : `stacks-v3-r16` (lora_B=0), `lora-qwen36-35b-hybrid` (vide), `stack-01-chat-fr-v2`, `qwen35-122b-macport`, `qwen35-35b-opus-{14k-v1,final}` (config-only). Dry-run par défaut, libère ~15.7 GB. |

## Usage

### `train_monitor_tui.py` — monitor 122B

Sur Studio :
```bash
.venv/bin/python tools/train_monitor_tui.py
```

Depuis GrosMac (TTY requis) :
```bash
ssh -t studio "cd ailiance-mac-tuner && .venv/bin/python tools/train_monitor_tui.py"
```

Options principales :
- `--log PATH` : log explicite (sinon dernier `train-*.log` sous `--root`).
- `--memcsv PATH` : memcsv explicite (sinon dernier `memcsv-*.csv`).
- `--root DIR` : base logs (defaut `logs/122b-macport`).
- `--total-iters N` : total iters pour la progress bar (defaut 3000).
- `--refresh SEC` : intervalle rafraichissement UI (defaut 2.0s).

### `archive_dead_artifacts.sh` — cleanup

```bash
# Sur Studio (où les vraies cibles existent) :
ssh studio "cd ailiance-mac-tuner && bash tools/archive_dead_artifacts.sh"             # dry-run par défaut
ssh studio "cd ailiance-mac-tuner && bash tools/archive_dead_artifacts.sh --execute"   # archive vraiment

# Pour libérer définitivement :
ssh studio "rm -rf ~/ailiance-mac-tuner/_archive/<date>"
```

Sécurités intégrées :
- Refuse si une cible est ouverte par un process (`lsof`)
- Logue chaque opération dans `_archive/<date>/archive.log`
- Move (pas suppression) — restaurable via `mv _archive/<date>/<target> <target>`
- Affiche tailles + nombre de fichiers avant action
