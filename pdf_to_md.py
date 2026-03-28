# pdf_to_md.py
import time
import re
from pathlib import Path

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

CONVERSIONS = [
    # (input_folder, output_folder, force_pymupdf)
    ("data/pdf/AI Reglementation/", "data/md/AI Reglementation/", False),
    ("data/pdf/arxiv/", "data/md/arxiv/", True),
    ("data/pdf/technical_guide/", "data/md/technical_guide/", False),
]

DOCLING_PAGES_THRESHOLD = 50  # au-delà : on force PyMuPDF


def clean_markdown(txt: str) -> str:
    txt = re.sub(r" {2,}", " ", txt)
    txt = re.sub(r"\t+", " ", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    txt = re.sub(r"- \n", "", txt)
    return txt.strip()


def pdf_has_heavy_images(pdf_path: Path, pixel_threshold: int = 50_000_000) -> bool:
    """Retourne True si le PDF contient des images très lourdes."""
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        for page in doc:
            for img in page.get_images():
                xref = img[0]
                image = doc.extract_image(xref)
                w = image.get("width", 0)
                h = image.get("height", 0)
                if w * h > pixel_threshold:
                    doc.close()
                    return True
        doc.close()
        return False
    except Exception:
        return False


# ─────────────────────────────────────────
# CONVERSION PYMUPDF (rapide, fallback)
# ─────────────────────────────────────────

def convert_with_pymupdf(pdf_path: Path, md_path: Path) -> bool:
    try:
        import fitz  # pip install pymupdf

        doc = fitz.open(str(pdf_path))
        pages = []
        for page in doc:
            pages.append(page.get_text())
        text = "\n\n".join(pages)
        text = clean_markdown(text)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(text, encoding="utf-8")
        print(f" {pdf_path.name} — PyMuPDF ({len(doc)} pages)")
        return True
    except Exception as e:
        print(f" {pdf_path.name} — PyMuPDF échoué : {e}")
        return False


# ─────────────────────────────────────────
# CONVERSION DOCLING (qualité)
# ─────────────────────────────────────────

def convert_with_docling(converter, pdf_path: Path, md_path: Path) -> bool:
    try:
        result = converter.convert(str(pdf_path))
        markdown = result.document.export_to_markdown()
        markdown = clean_markdown(markdown)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(markdown, encoding="utf-8")
        print(f" {pdf_path.name} — Docling")
        return True
    except Exception as e:
        print(f" {pdf_path.name} — Docling échoué ({e}), fallback PyMuPDF...")
        return convert_with_pymupdf(pdf_path, md_path)


# ─────────────────────────────────────────
# COMPTER LES PAGES (rapide)
# ─────────────────────────────────────────

def count_pdf_pages(pdf_path: Path) -> int:
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        n = len(doc)
        doc.close()
        return n
    except Exception:
        # inconnu → laisse Docling essayer
        return 0


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

if __name__ == "__main__":
    from docling.document_converter import DocumentConverter

    print("Initialisation de Docling...")
    converter = DocumentConverter()

    total_ok = 0
    total_err = 0

    for input_dir, output_dir, force_pymupdf in CONVERSIONS:
        input_path = Path(input_dir)
        output_path = Path(output_dir)

        if not input_path.exists():
            print(f"\n⚠ Dossier introuvable : {input_path}")
            continue

        pdf_files = list(input_path.rglob("*.pdf"))
        print(f"\n{'=' * 55}")
        print(f" {input_path} — {len(pdf_files)} fichiers PDF")
        print(f"{'=' * 55}")

        for pdf in pdf_files:
            md_out = output_path / f"{pdf.stem}.md"

            if md_out.exists():
                print(f" {pdf.name} — déjà converti, ignoré")
                total_ok += 1
                continue

            nb_pages = count_pdf_pages(pdf)
            print(f" {pdf.name} ({nb_pages} pages)...")
            start_t = time.time()

            if force_pymupdf or nb_pages > DOCLING_PAGES_THRESHOLD or pdf_has_heavy_images(pdf):
                print(" → PyMuPDF direct")
                ok = convert_with_pymupdf(pdf, md_out)
            else:
                ok = convert_with_docling(converter, pdf, md_out)

            duration = round(time.time() - start_t, 2)
            print(f" ({duration}s)")

            if ok:
                total_ok += 1
            else:
                total_err += 1

    print(f"\n{'=' * 55}")
    print(f"Conversion terminée : {total_ok} succès, {total_err} erreurs")