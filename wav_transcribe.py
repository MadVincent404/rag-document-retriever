# wav_index_from_json.py — uniquement embeddings + ChromaDB, pas de Whisper
import json
from pathlib import Path

from tqdm import tqdm
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

CHROMA_DB_DIR = "./data/chroma_db"
EMB_MODEL_NAME = "nomic-embed-text"
TRANSCRIPTS_DIR = "./data/wav/transcripts/"


def segments_to_docs_with_title(segments, wav_path: Path, duration: float):
    """
    Regroupe les segments (~60s) et ajoute le titre de la vidéo dans le premier chunk.
    """
    import re

    # "YOLOv1 from Scratch [n9_XyCGr-MI].wav" -> "YOLOv1 from Scratch"
    title = re.sub(r"\s*\[[^\]]+\]$", "", wav_path.stem).strip()

    docs = []
    chunk = []
    chunk_start = 0.0
    is_first_chunk = True

    for seg in segments:
        chunk.append(seg)
        if seg["end"] - chunk_start >= 60:
            text = " ".join(s["text"] for s in chunk)
            chunk_end = chunk[-1]["end"]

            if is_first_chunk:
                text = f"Video title: {title}\n\n{text}"
                is_first_chunk = False

            docs.append(
                Document(
                    page_content=f"[Video: {title}]\n{text}",
                    metadata={
                        "source": str(wav_path),
                        "filename": wav_path.name,
                        "video_title": title,
                        "type": "audio_transcript",
                        "category": "audio_transcript",
                        "langue": "en",
                        "duree_totale_s": duration,
                        "debut_s": chunk_start,
                        "fin_s": chunk_end,
                        "timestamp": f"{int(chunk_start // 60):02d}:{int(chunk_start % 60):02d}"
                        f" → {int(chunk_end // 60):02d}:{int(chunk_end % 60):02d}",
                    },
                )
            )
            chunk = []
            chunk_start = seg["end"]

    if chunk:
        text = " ".join(s["text"] for s in chunk)
        chunk_end = chunk[-1]["end"]
        if is_first_chunk:
            text = f"Video title: {title}\n\n{text}"
        docs.append(
            Document(
                page_content=f"[Video: {title}]\n{text}",
                metadata={
                    "source": str(wav_path),
                    "filename": wav_path.name,
                    "video_title": title,
                    "type": "audio_transcript",
                    "category": "audio_transcript",
                    "langue": "en",
                    "duree_totale_s": duration,
                    "debut_s": chunk_start,
                    "fin_s": chunk_end,
                    "timestamp": f"{int(chunk_start // 60):02d}:{int(chunk_start % 60):02d}"
                    f" → {int(chunk_end // 60):02d}:{int(chunk_end % 60):02d}",
                },
            )
        )
    return docs


if __name__ == "__main__":
    print("Étape 2 : Indexation JSON → ChromaDB (Ollama doit être démarré)")

    transcripts_dir = Path(TRANSCRIPTS_DIR)
    if not transcripts_dir.exists():
        print("Aucun transcript trouvé. Lance d'abord le script de transcription.")
        raise SystemExit(0)

    emb = OllamaEmbeddings(model=EMB_MODEL_NAME)
    vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=emb)

    # Fichiers déjà indexés
    try:
        all_meta = vector_db._collection.get(include=["metadatas"])
        already_indexed = {
            m.get("filename")
            for m in all_meta["metadatas"]
            if m.get("category") == "audio_transcript"
        }
        print(f" {len(already_indexed)} fichiers déjà dans ChromaDB")
    except Exception:
        already_indexed = set()

    all_chunks = []
    json_files = sorted(transcripts_dir.glob("*.json"))
    print(f" {len(json_files)} transcripts JSON trouvés")

    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data["filename"] in already_indexed:
            print(f" ↷ {data['filename']} — déjà indexé")
            continue

        print(f" + {data['filename']}")
        wav_path = Path("./data/wav/") / data["filename"]
        docs = segments_to_docs_with_title(
            data["segments"],
            wav_path,
            data["duree"],
        )
        all_chunks.extend(docs)

    if not all_chunks:
        print("Rien de nouveau à indexer.")
        raise SystemExit(0)

    print(f"\nTotal à vectoriser : {len(all_chunks)} chunks.")
    for i in tqdm(range(0, len(all_chunks), 16), desc="Sauvegarde ChromaDB"):
        vector_db.add_documents(all_chunks[i : i + 16])

    print(f"\nBase mise à jour ! {vector_db._collection.count()} chunks total.")