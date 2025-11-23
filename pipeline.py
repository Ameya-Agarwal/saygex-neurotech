import os
import subprocess

os.makedirs("./csvs", exist_ok=True)
os.makedirs("./plots", exist_ok=True)
os.makedirs("./json", exist_ok=True)

files_to_run = [
    "data_cleanup.py",
    "tumor_selection.py",
    "data_processing.py",
    "predictive_models.py",]

for f in files_to_run:
    print(f"\n=== RUNNING {f} ===\n")
    subprocess.run(["python", f], check=True)

