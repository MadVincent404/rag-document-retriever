import hashlib
import json
from pathlib import Path

from tqdm import tqdm
from faster_whisper import WhisperModel
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# --- CONFIG ---
WAV_DIR = "./data/wav/"
CHROMA_DB_DIR = "./data/chroma_db"
EMB_MODEL_NAME = "nomic-embed-text"
WHISPER_NAME = "small"  
FORCED_LANG = "en"
CACHE_PATH = "./data/wav/.transcription_cache.json"
TRANSCRIPTS_DIR = "./data/wav/transcripts/"  # stockage JSON


# ─────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────

def load_audio_cache():
    cache_file = Path(CACHE_PATH)
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_audio_cache(cache: dict) -> None:
    cache_file = Path(CACHE_PATH)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def file_md5(path: Path) -> str:
    """MD5 du fichier pour détecter les modifications."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


# ─────────────────────────────────────────
# TRANSCRIPTION
# ─────────────────────────────────────────

def transcribe_wav_file(model: WhisperModel, wav_path: Path, langue=FORCED_LANG):
    """
    Retourne une liste de segments avec horodatage :
    [{"start": 0.0, "end": 4.2, "text": "Bonjour..."}, ...]
    """
    segments, info = model.transcribe(
        str(wav_path),
        beam_size=5,
        language=langue,
        vad_filter=True,  # filtre les silences
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
    )

    out_segments = []
    for seg in segments:
        out_segments.append(
            {
                "start": round(seg.start, 2),
                "end": round(seg.end, 2),
                "text": seg.text.strip(),
            }
        )

    return out_segments, info.language, round(info.duration, 1)


def segments_to_docs(segments, wav_path: Path, detected_lang: str, duration: float):
    """
    Convertit les segments Whisper en Documents LangChain.
    Stratégie : regroupe les segments par fenêtres ~60s pour des chunks cohérents.
    """
    docs = []
    win_seconds = 60

    current_chunk = []
    chunk_start = 0.0

    for seg in segments:
        current_chunk.append(seg)
        chunk_duration = seg["end"] - chunk_start

        if chunk_duration >= win_seconds:
            text = " ".join(s["text"] for s in current_chunk)
            chunk_end = current_chunk[-1]["end"]

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(wav_path),
                        "filename": wav_path.name,
                        "type": "audio_transcript",
                        "category": "audio_transcript",
                        "langue": detected_lang or FORCED_LANG,
                        "duree_totale_s": duration,
                        "debut_s": chunk_start,
                        "fin_s": chunk_end,
                        "timestamp": f"{int(chunk_start // 60):02d}:{int(chunk_start % 60):02d}"
                        f" → {int(chunk_end // 60):02d}:{int(chunk_end % 60):02d}",
                    },
                )
            )
            current_chunk = []
            chunk_start = seg["end"]

    # Dernier chunk
    if current_chunk:
        text = " ".join(s["text"] for s in current_chunk)
        chunk_end = current_chunk[-1]["end"]
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(wav_path),
                    "filename": wav_path.name,
                    "type": "audio_transcript",
                    "category": "audio_transcript",
                    "langue": detected_lang or FORCED_LANG,
                    "duree_totale_s": duration,
                    "debut_s": chunk_start,
                    "fin_s": chunk_end,
                    "timestamp": f"{int(chunk_start // 60):02d}:{int(chunk_start % 60):02d}"
                    f" → {int(chunk_end // 60):02d}:{int(chunk_end % 60):02d}",
                },
            )
        )

    return docs


# ─────────────────────────────────────────
# PIPELINE AUDIO
# ─────────────────────────────────────────

def process_wav_folder():
    print("\nTraitement des fichiers Audio (.wav)...")

    wav_files = list(Path(WAV_DIR).rglob("*.wav"))
    if not wav_files:
        print("Aucun fichier .wav trouvé.")
        return []

    cache = load_audio_cache()

    files_to_run: list[tuple[Path, str]] = []
    for wav_path in wav_files:
        h = file_md5(wav_path)
        if cache.get(str(wav_path)) == h:
            print(f" ↷ {wav_path.name} — déjà transcrit, ignoré")
        else:
            files_to_run.append((wav_path, h))

    if not files_to_run:
        print("Tous les fichiers sont déjà dans le cache.")
        return []

    print(f"Chargement de Whisper ({WHISPER_NAME}) dans la VRAM...")
    model = WhisperModel(WHISPER_NAME, device="cuda", compute_type="float16")

    transcripts_dir = Path(TRANSCRIPTS_DIR)
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    all_docs = []
    for wav_path, hash_md5 in tqdm(files_to_run, desc="Transcription"):
        try:
            segments, lang, duration = transcribe_wav_file(model, wav_path)
            print(f" ✓ {wav_path.name} — {duration}s, {len(segments)} segments")

            docs = segments_to_docs(segments, wav_path, lang, duration)
            all_docs.extend(docs)

            transcript_file = transcripts_dir / f"{wav_path.stem}.json"
            with open(transcript_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "filename": wav_path.name,
                        "segments": segments,
                        "duree": duration,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )

            cache[str(wav_path)] = hash_md5

        except Exception as e:
            print(f" ✗ {wav_path.name} — erreur : {e}")

    save_audio_cache(cache)
    return all_docs


if __name__ == "__main__":
    print("Démarrage de l'ingestion Audio...")

    # Étape 1 : transcription éventuelle
    new_chunks = process_wav_folder()

    # Étape 2 : indexation dans ChromaDB depuis tous les JSON
    print("\nChargement des transcriptions depuis le disque...")
    emb = OllamaEmbeddings(model=EMB_MODEL_NAME)
    vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=emb)

    try:
        all_meta = vector_db._collection.get(include=["metadatas"])
        already_indexed = {
            m.get("filename")
            for m in all_meta["metadatas"]
            if m.get("category") == "audio_transcript"
        }
        print(f" {len(already_indexed)} fichiers audio déjà dans ChromaDB")
    except Exception:
        already_indexed = set()

    transcripts_dir = Path(TRANSCRIPTS_DIR)
    chunks_to_index = []

    if transcripts_dir.exists():
        for json_file in sorted(transcripts_dir.glob("*.json")):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data["filename"] in already_indexed:
                print(f" ↷ {data['filename']} — déjà indexé")
                continue

            print(f" + {data['filename']}")
            wav_path = Path(WAV_DIR) / data["filename"]
            docs = segments_to_docs(
                data["segments"],
                wav_path,
                FORCED_LANG,
                data["duree"],
            )
            chunks_to_index.extend(docs)

    all_chunks = new_chunks + chunks_to_index

    if not all_chunks:
        print("Rien de nouveau à indexer.")
        raise SystemExit(0)

    print(f"\nTotal à vectoriser : {len(all_chunks)} chunks.")

    batch_size = 16
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Sauvegarde ChromaDB"):
        vector_db.add_documents(all_chunks[i : i + batch_size])

    print(
        f"\nBase mise à jour ! {vector_db._collection.count()} chunks total dans ChromaDB"
    )