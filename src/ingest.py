import json
import re
from pathlib import Path

import pandas as pd
import pymupdf

PDF_DIR = Path("data/pdfs")
META_CSV = Path("papers_metadata.csv")
OUT_FILE = Path("output/chunks.json")

CHUNK_WORDS = 350   # sekitar 450-500 token
OVERLAP_WORDS = 60
KEEP_FULL = {"P08"}


def clean(text: str) -> str:
    # gabungkan kata yang terpotong di akhir baris
    text = text.replace("-\n", "")
    text = re.sub(r"\s+", " ", text)        # rapikan spasi dan baris baru
    return text.strip()


def read_pdf(path: Path, trim_refs: bool = True):
    """Return list of (page_number, text). Berhenti saat masuk daftar pustaka."""
    pages = []
    doc = pymupdf.open(path)
    total = len(doc)
    for i, page in enumerate(doc):
        raw = page.get_text("text")
        if trim_refs and i > total * 0.4:
            m = re.search(r"\n\s*(references|bibliography)\s*\n",
                          raw, re.IGNORECASE)
            if m:
                raw = raw[: m.start()]
                if clean(raw):
                    pages.append((i + 1, clean(raw)))
                break
        text = clean(raw)
        if text:
            pages.append((i + 1, text))
    doc.close()
    return pages


def chunk_pages(pages):
    """Pecah teks per paper menjadi chunk berisi kata, simpan halaman awal dan akhir."""
    words = []  # list of (word, page)
    for page_no, text in pages:
        for w in text.split():
            words.append((w, page_no))

    chunks = []
    step = CHUNK_WORDS - OVERLAP_WORDS
    for start in range(0, len(words), step):
        part = words[start: start + CHUNK_WORDS]
        if len(part) < 50:  # buang sisa yang terlalu pendek
            break
        chunks.append(
            {
                "text": " ".join(w for w, _ in part),
                "page_start": part[0][1],
                "page_end": part[-1][1],
            }
        )
    return chunks


def main():
    meta = {}
    if META_CSV.exists():
        df = pd.read_csv(META_CSV).fillna("")
        meta = {row["file"]: row.to_dict() for _, row in df.iterrows()}

    OUT_FILE.parent.mkdir(exist_ok=True)
    all_chunks = []

    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        paper_id = pdf.name.split("_")[0]
        info = meta.get(pdf.name, {})
        pages = read_pdf(pdf, trim_refs=paper_id not in KEEP_FULL)
        chunks = chunk_pages(pages)
        n_words = sum(len(t.split()) for _, t in pages)

        for idx, c in enumerate(chunks):
            c.update(
                {
                    "chunk_id": f"{paper_id}_c{idx:03d}",
                    "paper_id": paper_id,
                    "file": pdf.name,
                    "judul": info.get("judul", ""),
                    "penulis": info.get("penulis", ""),
                    "tahun": info.get("tahun", ""),
                }
            )
        all_chunks.extend(chunks)

        flag = "  <-- CEK: teks sangat sedikit, mungkin PDF hasil scan" if n_words < 1000 else ""
        print(
            f"{paper_id}: {len(pages)} halaman, {n_words} kata, {len(chunks)} chunk{flag}")

    OUT_FILE.write_text(json.dumps(
        all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nTotal {len(all_chunks)} chunk disimpan ke {OUT_FILE}")


if __name__ == "__main__":
    main()
