# Reproduce every result of PS-8 end to end (Windows PowerShell).
# Usage:  .\run_all.ps1            (assumes the virtual environment in .venv is set up, see README)
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"

& $py scripts\download_tawos.py --out data\raw\TAWOS.sql.zip      # 1. TAWOS dump (637 MB, md5-checked)
& $py -m src.extract_tawos                                          # 2. Issue / Project tables -> data/interim
& $py -m src.preprocess                                             # 3. filtering, cleaning, sentence split
& $py -m src.export_splits                                          #    exact dev/test partitions per pair -> results/splits
& $py -m src.embed_sbert                                            # 4. SBERT sentence embeddings (GPU ~6 min)
& $py -m src.run_experiments --tag main                             # 5. 39 pairs, 5-fold CV + held-out test
& $py -m src.run_experiments --tag strict --strict                  # 6. sensitivity: strict cross-project chronology
& $py -m src.analyze_results --tag main --compare strict            # 7. tables, statistics, figures
