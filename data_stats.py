import os
from pathlib import Path
import json
import fitz  

PDF_ROOT = Path("./data/pdf")
WAV_ROOT = Path("./data/wav")


def count_pdf_stats():
    """Compte PDFs et pages exactes."""
    subdirs = ["AI Reglementation", "arxiv", "technical_guide"]
    total_pdfs = 0
    total_pages = 0

    print("\nStats PDFs :")
    for subdir in subdirs:
        dir_path = PDF_ROOT / subdir
        if not dir_path.exists():
            print(f"  {subdir:20} : manquant")
            continue

        pdfs = list(dir_path.glob("*.pdf"))
        total_pdfs += len(pdfs)
        pages_in_dir = 0

        print(f"  {subdir:20} : {len(pdfs)} PDFs")
        for pdf_path in pdfs:
            try:
                doc = fitz.open(pdf_path)
                pages_in_dir += len(doc)
                doc.close()
            except Exception as e:
                print(f"    {pdf_path.name} : erreur ({e})")

        total_pages += pages_in_dir
        print(f"    → {pages_in_dir} pages")

    return total_pdfs, total_pages


def count_audio_stats():
    """Heures audio + cache."""
    subdirs = ["LLM", "Pytorch", "RAG", "RAG_Scratch"]
    total_wavs = 0
    total_seconds = 0

    print("\nStats Audio :")
    for subdir in subdirs:
        dir_path = WAV_ROOT / subdir
        if not dir_path.exists():
            print(f"  {subdir:20} : manquant")
            continue

        wavs = list(dir_path.glob("*.wav"))
        total_wavs += len(wavs)
        seconds_in_dir = 0

        print(f"  {subdir:20} : {len(wavs)} WAV")
        for wav_path in wavs:
            try:
                size_bytes = wav_path.stat().st_size
                # Estimation 44.1kHz stereo 16bit
                duration_s = size_bytes / (44100 * 4)
                seconds_in_dir += duration_s
            except:
                pass

        total_seconds += seconds_in_dir
        print(f"    → {seconds_in_dir/3600:.1f}h")

    # Cache transcription
    cache_path = WAV_ROOT / ".transcription_cache.json"
    cache_count = 0
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
            cache_count = len(cache)
            print(f"\nCache transcription : {cache_count}/{total_wavs} OK")
        except:
            print(f"\nCache transcription corrompu")

    return total_wavs, total_seconds / 3600, cache_count


if __name__ == "__main__":
    print("=== STATS DATA RAG ===\n")

    pdf_count = pdf_pages = 0
    if PDF_ROOT.exists():
        pdf_count, pdf_pages = count_pdf_stats()
    else:
        print("data/pdf/ manquant → lance pipeline_full.py")

    wav_count = audio_h = cache_ok = 0
    if WAV_ROOT.exists():
        wav_count, audio_h, cache_ok = count_audio_stats()
    else:
        print("data/wav/ manquant → ajoute des WAV")

    print("\n" + "=" * 50)
    print(f"RÉSUMÉ GLOBAL")
    print(f"{pdf_count} PDFs ({pdf_pages} pages)")
    print(f"{wav_count} WAV ({audio_h:.1f}h | {cache_ok} transcrits)")
    print("=" * 50)