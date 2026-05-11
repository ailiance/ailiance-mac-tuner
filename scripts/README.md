# scripts/

Scripts bash et Python pour le pipeline ailiance-mac-tuner : telechargement,
conversion, training, distillation, export GGUF, monitoring.

Voir `CLAUDE.md` dans ce dossier pour les conventions internes.

## Training

| Fichier | Role |
|---|---|
| `train_mlx.py` | Entry-point LoRA fine-tuning MLX (Studio M3 Ultra / M4 Pro). Wrappe `mlx_lm.lora` avec options projet. |
| `train_122b_macport.sh` | Lance le training Qwen3.5-122B-A10B-BF16 profil mac-port (logs + watchdog memoire). |
| `pipeline_35b.sh` | Pipeline complet 35B : merge donnees -> fine-tune Opus -> fuse -> export GGUF Q4. Sous-commandes : `merge|train|fuse|gguf|all|status`. |
| `pipeline_122b.sh` | Pipeline complet 122B MoE Opus-v3 : `distill|merge|train|fuse|gguf|all|status`. |
| `watchdog_mem.sh` | Surveille RSS / swap / peak Metal pendant un training, kill si swap > 80 Go sur 10 echantillons. CSV dans `logs/`. |
| `monitor.sh` | Monitoring de fond des pipelines (`start|stop|status|tail`). |

## Distillation / generation de donnees

| Fichier | Role |
|---|---|
| `generate_data.py` | Generation synthetique de traces de raisonnement via teachers MLX. |
| `generate_data_cpu.py` | Equivalent CPU via llama.cpp (tourne en parallele du training GPU Metal). |
| `generate_cpu.sh` | Wrapper bash autour de `generate_data_cpu.py`. |
| `distill_generate.py` | Genere des traces via Mistral Large 123B fusionne (teacher -> student 35B). |
| `distill_ane.py` | Distillation via ANE (import direct ANEMLL, 8x vs subprocess, scoring parallele). |
| `distill_mlxvlm.py` | Distillation rapide via mlx-vlm natif (~45 tok/s) a partir de Qwen3.5-35B-A3B 4bit. |
| `ane_inference.py` | Inference ANE optimisee (import direct ANEMLL, scoring parallele 3 instances, batch prefill). |

## Datasets

| Fichier | Role |
|---|---|
| `download_datasets.sh` | Telecharge les datasets reasoning Opus 4.6 depuis HF. |
| `download_teachers.sh` | Telecharge les modeles teachers (DeepSeek R1, Mistral Large, Qwen3...). |
| `download_gguf.sh` | Telecharge des modeles GGUF pour llama.cpp. |
| `convert_datasets.py` | Normalise les datasets vers le format chat JSONL `{"messages": [...]}`. |
| `prepare_combined_dataset.sh` | Pipeline download + convert + merge pour le dataset combine final. |

## Conversion / export

| Fichier | Role |
|---|---|
| `convert_to_mlx.sh` | Convertit un modele HF au format MLX bf16. |
| `convert_ane.sh` | Convertit un Qwen3 dense pour Apple Neural Engine via ANEMLL (MoE non supporte a ce jour). |
| `merge_lora.py` | Fusionne un adapter LoRA dans les poids base (wrapper `mlx_lm.fuse`). |
| `convert_gguf.py` | Convertit le modele fusionne en GGUF + quantization (via `llama.cpp/convert_hf_to_gguf.py`). |
| `export_gguf.sh` | Pipeline train student 35B + export GGUF : `train|fuse|convert|all`. |
