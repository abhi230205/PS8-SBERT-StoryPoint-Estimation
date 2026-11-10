#!/usr/bin/env bash
# Reproduce every result of PS-8 end to end (Linux / macOS / Git Bash).
# Usage:  bash run_all.sh   (activate the virtual environment first, see README)
set -euo pipefail
PY=${PYTHON:-python}

$PY scripts/download_tawos.py --out data/raw/TAWOS.sql.zip   # 1. TAWOS dump (637 MB, md5-checked)
$PY -m src.extract_tawos                                     # 2. Issue / Project tables -> data/interim
$PY -m src.preprocess                                        # 3. filtering, cleaning, sentence split
$PY -m src.export_splits                                     #    exact dev/test partitions per pair -> results/splits
$PY -m src.embed_sbert                                       # 4. SBERT sentence embeddings
$PY -m src.run_experiments --tag main                        # 5. 39 pairs, 5-fold CV + held-out test
$PY -m src.run_experiments --tag strict --strict             # 6. sensitivity: strict cross-project chronology
$PY -m src.analyze_results --tag main --compare strict       # 7. tables, statistics, figures
