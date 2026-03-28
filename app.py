import os
import base64
import re
import uuid
from io import BytesIO

import streamlit as st
import streamlit.components.v1 as components

import PyPDF2
from pathlib import Path

from main import (
    route_query,
    run_rag_query,
    save_feedback,
    sanitize_user_input,
)

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="RAG Multi-Source",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# ARBORESCENCE DYNAMIQUE
# ─────────────────────────────────────────

def show_tree_view(active_cat: str | None = None):
    highlight_style = (
        "background-color: #4CAF50; color: white; padding: 2px 6px; "
        "border-radius: 4px; font-weight: bold;"
    )
    targets = {
        "DOC_LEGAL": "AI Reglementation",
        "DOC_RESEARCH": "arxiv",
        "DOC_TECHNICAL": "technical_guide",
        "DOC_AUDIO": "wav",
        "CODE": "code",
    }
    target_folder = targets.get(active_cat, "")

    tree_html = f"""
    <div style="font-family: monospace; font-size: 13px;">
      <div>data/</div>
      <div style="margin-left: 16px;">
        <div>pdf/</div>
        <div style="margin-left: 16px;">
          <div>{'<span style="' + highlight_style + '">AI Reglementation</span>' if target_folder == 'AI Reglementation' else 'AI Reglementation'}/</div>
          <div>{'<span style="' + highlight_style + '">arxiv</span>' if target_folder == 'arxiv' else 'arxiv'}/</div>
          <div>{'<span style="' + highlight_style + '">technical_guide</span>' if target_folder == 'technical_guide' else 'technical_guide'}/</div>
        </div>
        <div>md/</div>
        <div style="margin-left: 16px;">
          <div>{'<span style="' + highlight_style + '">AI Reglementation</span>' if target_folder == 'AI Reglementation' else 'AI Reglementation'}/</div>
          <div>{'<span style="' + highlight_style + '">arxiv</span>' if target_folder == 'arxiv' else 'arxiv'}/</div>
          <div>{'<span style="' + highlight_style + '">technical_guide</span>' if target_folder == 'technical_guide' else 'technical_guide'}/</div>
        </div>
        <div>chroma_db/</div>
        <div>wav/</div>
        <div style="margin-left: 16px;">
          <div>{'<span style="' + highlight_style + '">wav</span>' if target_folder == 'wav' else 'wav'}/</div>
          <div>transcripts/</div>
        </div>
        <div>code/</div>
      </div>
    </div>
    """
    st.markdown(tree_html, unsafe_allow_html=True)


# ─────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────

def read_pdf_text(uploaded_file) -> str:
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text_pages = []
        for page in reader.pages:
            text_pages.append(page.extract_text() or "")
        text = "\n".join(text_pages)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    except Exception:
        return ""


def get_file_download_link(content: str, filename: str, label: str) -> None:
    b = content.encode("utf-8")
    b64 = base64.b64encode(b).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{label}</a>'
    st.markdown(href, unsafe_allow_html=True)


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.header("Config requête")

    default_cat = "DOC_RESEARCH"
    st.caption("La catégorie finale est décidée par le routeur sémantique.")
    show_tree = st.checkbox("Afficher arborescence data", value=False)

    st.markdown("---")
    st.caption("Upload optionnel d'un PDF (pré-lecture, pas indexé automatiquement).")
    uploaded_pdf = st.file_uploader("PDF (optionnel)", type=["pdf"])

    st.markdown("---")
    show_debug = st.checkbox("Afficher debug RAG", value=False)

# ─────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────

col_left, col_right = st.columns([2, 1])

with col_left:
    st.title("RAG multi-sources (docs + audio)")

    user_question = st.text_area(
        "Ta question",
        height=120,
        placeholder="Pose une question sur les docs (AI Act, arxiv, guides techniques) ou les vidéos audio...",
    )

    ask_btn = st.button("Lancer la recherche")

with col_right:
    if show_tree:
        st.subheader("Arborescence data/")
        show_tree_view()

    if uploaded_pdf is not None:
        st.subheader("Preview PDF uploadé")
        pdf_text = read_pdf_text(uploaded_pdf)
        if pdf_text:
            st.text_area("Texte extrait (preview)", pdf_text[:4000], height=200)
            get_file_download_link(pdf_text, "pdf_extrait.txt", "Télécharger texte brut")
        else:
            st.info("Impossible d'extraire le texte de ce PDF.")

# ─────────────────────────────────────────
# HANDLER PRINCIPAL
# ─────────────────────────────────────────

if ask_btn:
    try:
        safe_question = sanitize_user_input(user_question)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    with st.spinner("Routing de la question..."):
        cat, cat_filter = route_query(safe_question)

    st.write(f"Catégorie détectée : `{cat}`")

    with st.spinner("RAG en cours..."):
        answer, doc_source, debug_info = run_rag_query(
            safe_question,
            filtre=cat_filter,
        )

    st.markdown("### Réponse")
    st.write(answer)

        # Aperçu ressource liée (YouTube / doc) si dispo
    if doc_source is not None:
        src = doc_source.metadata.get("source", "")
        filename = doc_source.metadata.get("filename", "")
        video_title = doc_source.metadata.get("video_title", "")

        # --- Cas audio / vidéo YouTube ---
        if doc_source.metadata.get("category") == "audio_transcript":
            # On tente d'extraire l'ID YouTube depuis le nom de fichier
            # ex: "YOLOv1 from Scratch [n9_XyCGr-MI].wav"
            m = re.search(r"\[([A-Za-z0-9_-]{6,})\]", filename)
            yt_id = m.group(1) if m else None

            if yt_id:
                st.markdown("#### Vidéo liée")
                yt_url = f"https://www.youtube.com/embed/{yt_id}"
                components.iframe(
                    yt_url,
                    height=360,
                    width=640,
                )
                if video_title:
                    st.caption(f"Source vidéo : {video_title}")
            else:
                st.caption("Vidéo audio liée, mais impossible de retrouver l'URL YouTube.")

        # --- Cas PDF / docs locaux ---
        elif src.endswith(".pdf"):
            st.markdown("#### Aperçu du PDF source")
            try:
                pdf_path = Path(src)
                if pdf_path.exists():
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                    # On emballe les premières pages en iframe via base64
                    import base64

                    b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
                    pdf_display = f"""
                    <iframe src="data:application/pdf;base64,{b64_pdf}" 
                            width="700" height="500" type="application/pdf">
                    </iframe>
                    """
                    st.markdown(pdf_display, unsafe_allow_html=True)
                else:
                    st.caption(f"PDF source introuvable : {pdf_path}")
            except Exception as e:
                st.caption(f"Aperçu PDF impossible ({e})")

    # Feedback zone
    st.markdown("---")
    st.subheader("Feedback")
    col_fb1, col_fb2 = st.columns([1, 3])
    with col_fb1:
        thumb_choice = st.radio(
            "Réponse utile ?",
            options=["up", "down"],
            horizontal=True,
            index=0,
            format_func=lambda x: "Oui" if x == "up" else "Non",
        )
    with col_fb2:
        precision_txt = st.text_input(
            "Détail optionnel (ce qui manque, ce qui est bien...)", value=""
        )

    if st.button("Envoyer le feedback"):
        qid = str(uuid.uuid4())
        save_feedback(
            query_id=qid,
            question=safe_question,
            answer=answer,
            category=cat,
            thumbs=thumb_choice,
            precision=precision_txt,
            source_meta=(doc_source.metadata if doc_source else None),
        )
        st.success("Feedback enregistré.")

    # Debug zone
    if show_debug:
        st.markdown("---")
        st.subheader("Debug RAG")
        st.json(debug_info)
        if doc_source is not None:
            st.markdown("**Source principale**")
            st.write(doc_source.metadata)