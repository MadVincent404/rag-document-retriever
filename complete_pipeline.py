# pipeline_full.py
import subprocess
import sys
import shutil
from pathlib import Path
import os

# On se place dans le dossier du script
os.chdir(Path(__file__).parent)


def run_script(script_name: str) -> None:
    print(f"\n{'=' * 55}")
    print(f" Lancement script : {script_name}")
    print(f"{'=' * 55}")
    subprocess.run([sys.executable, script_name], check=True)


# ── ÉTAPE 0 : Nettoyage data ──────────────────────────────
print("\n=== NETTOYAGE DATA LOCALE ===")

folders_to_clean = [
    "./data/md",
    "./data/chroma_db",
]

for folder in folders_to_clean:
    path = Path(folder)
    if path.exists():
        shutil.rmtree(path)
        print(f" Supprimé : {folder}")

audio_cache = Path("./data/wav/.transcription_cache.json")
if audio_cache.exists():
    audio_cache.unlink()
    print(" Cache audio supprimé.")

# ── ÉTAPE 1 : PDF → Markdown ──────────────────────────────
run_script("pdf_to_md.py")

# ── ÉTAPE 2 : Markdown → ChromaDB ─────────────────────────
run_script("reindex.py")

# ── ÉTAPE 3 : WAV → ChromaDB (optionnel) ──────────────────
# run_script("wav_to_chroma_db.py")

print("\n=== Pipeline complet terminé. Lancer app.py ===")