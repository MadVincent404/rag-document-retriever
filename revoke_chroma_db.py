from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectordb   = Chroma(persist_directory="./data/chroma_db", embedding_function=embeddings)

all_data = vectordb._collection.get(include=["metadatas"])
ids_audio = [
    all_data["ids"][i]
    for i, m in enumerate(all_data["metadatas"])
    if m.get("category") == "audio_transcript"
]
print(f"Suppression de {len(ids_audio)} chunks audio...")
vectordb._collection.delete(ids=ids_audio)
print(f"Chunks restants : {vectordb._collection.count()}")