import json
import os
import re
import time
from pathlib import Path

import faiss
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

load_dotenv()

CHUNKS_FILE = Path("output/chunks.json")
INDEX_FILE = Path("output/faiss.index")
OUT_FILE = Path("output/extraction.json")
EMB_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gemini-3.1-flash-lite"
TOP_K = 3          # chunk per kolom
SLEEP_SEC = 10     # jeda antar paper

COLUMNS = {
    "masalah_tujuan": "What problem or objective does this paper address?",
    "subdomain_kesehatan": "Which healthcare domain or clinical task does it target (e.g. medical QA, EHR, decision support, patient chatbot)?",
    "retriever_embedding": "Which retriever, embedding model, or vector database is used?",
    "generator_llm": "Which large language model (generator) is used?",
    "strategi_chunking": "How are documents chunked or preprocessed (chunk size, overlap, splitting method)?",
    "dataset_knowledge_base": "Which datasets or knowledge base/corpus are used?",
    "metrik_evaluasi": "Which evaluation metrics are used?",
    "evaluasi_faithfulness": "Is hallucination, faithfulness, or factual grounding evaluated? How?",
    "bahasa_data": "Which language(s) is the data or system in?",
    "hasil_utama": "What are the main quantitative or qualitative results?",
    "keterbatasan": "What limitations does the paper state?",
    "future_work": "What future work does the paper suggest?",
}

PROMPT = """You extract information from ONE scientific paper for a literature gap matrix.
Below are several fields. Each field has its own question and its own retrieved context.
Answer each field ONLY from the context given for that field. Do not use outside knowledge.
If the context does not contain the answer, set "value" to "Not reported".

{sections}

Return ONLY a JSON object with exactly these keys: {keys}.
Each key maps to: {{"value": "concise answer in English, max 40 words", "evidence_pages": [page numbers as integers], "confidence": "high|medium|low"}}
"""


def parse_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(),
                  flags=re.MULTILINE).strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0) if m else text)


def ask_llm(client, prompt: str):
    cfg = types.GenerateContentConfig(response_mime_type="application/json")
    for attempt in range(4):
        try:
            t0 = time.time()
            resp = client.models.generate_content(
                model=LLM_MODEL, contents=prompt, config=cfg)
            out = parse_json(resp.text)
            print(f"   LLM selesai dalam {time.time() - t0:.0f} detik")
            return out
        except Exception as e:
            wait = 20 * (attempt + 1)
            print(
                f"   ! percobaan {attempt + 1} gagal ({str(e)[:150]}), tunggu {wait}s")
            time.sleep(wait)
    return None


def main():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "isi_key_kamu":
        raise SystemExit("GEMINI_API_KEY belum diisi di file .env")
    client = genai.Client(api_key=api_key)

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    index = faiss.read_index(str(INDEX_FILE))
    model = SentenceTransformer(EMB_MODEL)

    paper_ids = sorted({c["paper_id"] for c in chunks})
    results = json.loads(OUT_FILE.read_text(
        encoding="utf-8")) if OUT_FILE.exists() else {}

    for pid in paper_ids:
        done = results.get(pid, {})
        if len(done) == len(COLUMNS) and all(v.get("value") != "ERROR" for v in done.values()):
            print(f"{pid}: sudah selesai, dilewati")
            continue

        print(f"\n== {pid} ==")
        allowed = {i for i, c in enumerate(chunks) if c["paper_id"] == pid}
        sections, retrieved = [], {}

        for col, question in COLUMNS.items():
            q = "Represent this sentence for searching relevant passages: " + question
            qv = model.encode([q], normalize_embeddings=True).astype("float32")
            _, ids = index.search(qv, len(chunks))
            top = [i for i in ids[0] if i in allowed][:TOP_K]
            retrieved[col] = [chunks[i]["chunk_id"] for i in top]
            ctx = "\n".join(
                f"[Page {chunks[i]['page_start']}-{chunks[i]['page_end']}] {chunks[i]['text']}" for i in top
            )
            sections.append(
                f"### FIELD: {col}\nQuestion: {question}\nContext:\n{ctx}\n")

        prompt = PROMPT.format(sections="\n".join(
            sections), keys=", ".join(COLUMNS))
        out = ask_llm(client, prompt)

        results[pid] = {}
        for col in COLUMNS:
            ans = (out or {}).get(col) or {"value": "ERROR",
                                           "evidence_pages": [], "confidence": "low"}
            ans["retrieved_chunks"] = retrieved[col]
            results[pid][col] = ans
            print(f" {col}: {str(ans.get('value'))[:70]}")

        OUT_FILE.write_text(json.dumps(
            results, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(SLEEP_SEC)

    print(f"\nSelesai. Hasil di {OUT_FILE}")


if __name__ == "__main__":
    main()
