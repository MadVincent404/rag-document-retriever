import os
import re
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_DB_DIR = "./data/chroma_db"
MD_ROOT_DIR = "./data/md"
EMB_MODEL_NAME = "nomic-embed-text"

emb = OllamaEmbeddings(model=EMB_MODEL_NAME)

FOLDER_CATEGORY = {
    "AI Reglementation": "legal",
    "arxiv": "research",
    "technical_guide": "technical",
}


def clean_md_text(txt: str) -> str:
    txt = re.sub(r" {2,}", " ", txt)
    txt = re.sub(r"\t+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = re.sub(r"- \n", "", txt)
    return txt.strip()


# ── Collecte tous les fichiers .md ─────────────────────────
md_files = []
for folder_name, doc_cat in FOLDER_CATEGORY.items():
    folder_path = os.path.join(MD_ROOT_DIR, folder_name)
    if not os.path.exists(folder_path):
        continue
    for fname in os.listdir(folder_path):
        if fname.endswith(".md"):
            full_path = os.path.join(folder_path, fname)
            md_files.append(
                {
                    "path": full_path,
                    "category": doc_cat,
                    "book_title": os.path.splitext(fname)[0].lower(),
                    "source": full_path.replace("\\", "/"),
                }
            )

print(f"{len(md_files)} fichiers .md trouvés.")

# ── Sources déjà indexées dans ChromaDB ────────────────────
vector_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=emb)

try:
    all_meta = vector_db._collection.get(include=["metadatas"])
    indexed_sources = {
        m.get("source", "").replace("\\", "/")
        for m in all_meta["metadatas"]
        if m.get("category") in ("legal", "research", "technical")
    }
    print(f"{len(indexed_sources)} sources déjà indexées dans ChromaDB.")
except Exception:
    indexed_sources = set()

# ── Filtre les nouveaux fichiers uniquement ─────────────────
new_files = [f for f in md_files if f["source"] not in indexed_sources]
print(f"{len(new_files)} nouveaux fichiers à indexer.")

if not new_files:
    print("Rien de nouveau à indexer.")
    raise SystemExit(0)

# ── Chargement + découpage ──────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
)

doc_chunks = []
for file_info in tqdm(new_files, desc="Chargement"):
    loader = TextLoader(file_info["path"], encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.page_content = clean_md_text(doc.page_content)
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["category"] = file_info["category"]
        chunk.metadata["book_title"] = file_info["book_title"]
    doc_chunks.extend(chunks)

print(f"\n{len(doc_chunks)} chunks à ajouter...")

# ── Ajoute sans supprimer l'existant ────────────────────────
batch_size = 32
for i in tqdm(range(0, len(doc_chunks), batch_size), desc="Indexation"):
    vector_db.add_documents(doc_chunks[i : i + batch_size])

print(f"Indexation terminée ! Total chunks : {vector_db._collection.count()}")