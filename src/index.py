import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = Path("output/chunks.json")
INDEX_FILE = Path("output/faiss.index")
# ringan, bagus untuk teks ilmiah berbahasa Inggris
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main():
    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    texts = [c["text"] for c in chunks]
    print(f"Memuat model {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Meng-embed {len(texts)} chunk ...")
    emb = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    emb = np.asarray(emb, dtype="float32")

    # cosine similarity (vektor sudah dinormalisasi)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, str(INDEX_FILE))
    print(
        f"Indeks berisi {index.ntotal} vektor, dimensi {emb.shape[1]}, disimpan ke {INDEX_FILE}")


if __name__ == "__main__":
    main()
