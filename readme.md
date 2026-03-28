# RAG Multi-Source (Docs PDF + Audio YouTube)

Application Streamlit RAG locale avec **Ollama** (embeddings), **ChromaDB** (vector store), et sources mixtes :  
- **Docs** : PDF → Markdown → ChromaDB  
- **Audio** : WAV YouTube → transcription → ChromaDB  
- **LLM** : Groq (via LangChain) + routeur sémantique  

## Prérequis
- **Python 3.11.9**  
- **Ollama** installé + modèles : `nomic-embed-text`  
- **Groq API key** (gratuit) : https://console.groq.com/keys  
- GPU recommandé (CUDA pour Whisper/Ollama)  

## Installation
```bash
python3.11 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Config Groq
mkdir -p .streamlit
echo 'GROQ_API_KEY = "gsk_..."' > .streamlit/secrets.toml
```

## Utilisation
```bash
# 1. Lance Ollama
ollama serve
ollama pull nomic-embed-text

# 2. Prépare les données (optionnel)
python pipeline_full.py    # PDF → ChromaDB
python wav_to_chroma_db.py # Audio → ChromaDB

# 3. Lance l'app
streamlit run app.py
```

**URLs** :  
- `http://localhost:8501` → App RAG  
- `http://localhost:8502` → Dashboard feedbacks  

## Scripts
| Script | Action |
|--------|--------|
| `pipeline_full.py` | Nettoie + PDF → ChromaDB |
| `wav_to_chroma_db.py` | Transcription + index audio |
| `wav_index_from_json.py` | Index JSON existants |
| `cleanup_audio_chunks.py` | Purge audio ChromaDB |

Frontend : Streamlit
Vector DB : ChromaDB
Embeddings : Ollama nomic-embed-text
LLM : Groq Llama3.3 70B
Audio : faster-whisper (CUDA)
PDF : Docling + PyMuPDF