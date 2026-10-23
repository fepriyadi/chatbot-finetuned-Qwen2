"""Cek ulang RERANK_THRESHOLD di lingkungan deployment (skor reranker bisa bergeser antar device).

    python -m legal_rag.calibrate

Sama dengan kalibrasi di notebook: ensemble tanpa HyDE, skor Top-1 pertanyaan dalam vs luar domain.
"""
from legal_rag.config import N_CANDIDATES, RERANK_THRESHOLD
from legal_rag.pipeline import LegalRAG

TEST_QUERY = "Saya staf admin, kemarin lembur 3 jam untuk beresin laporan. Apakah saya berhak dapat uang lembur?"

CALIBRATION_QUESTIONS = {
    "Berapa lama jangka waktu maksimal PKWT?": "dalam domain (formal)",
    "Bagaimana formula perhitungan upah minimum?": "dalam domain (formal)",
    "Berapa pesangon untuk pekerja yang di-PHK karena perusahaan pailit?": "dalam domain (formal)",
    TEST_QUERY: "dalam domain (percakapan)",
    "Siapa pemenang Piala Dunia 2022?": "luar domain",
    "Bagaimana cara membuat kue brownies yang lembut?": "luar domain",
    "Berapa harga Bitcoin hari ini?": "luar domain",
}


def main():
    rag = LegalRAG()
    retriever, _ = rag.build_hybrid_retriever(use_hyde=False)
    in_domain, out_domain = [], []
    for question, category in CALIBRATION_QUESTIONS.items():
        doc, score = rag.rerank(question, retriever.invoke(question)[:N_CANDIDATES], top_k=1)[0]
        (out_domain if category == "luar domain" else in_domain).append(score)
        print(f"{score:8.4f}  {category:26s} {doc.metadata['singkatan']} hal. {doc.metadata['page']:<4} {question[:60]}")

    in_domain_min, out_domain_max = min(in_domain), max(out_domain)
    print(f"\nSkor terendah dalam domain: {in_domain_min:.4f} | skor tertinggi luar domain: {out_domain_max:.4f}")
    if out_domain_max < RERANK_THRESHOLD < in_domain_min:
        print(f"RERANK_THRESHOLD = {RERANK_THRESHOLD} masih memisahkan kedua kelompok.")
    else:
        # titik tengah (skala log) kedua kelompok, sama dengan fallback di notebook
        suggested = round(float((max(out_domain_max, 1e-4) * in_domain_min) ** 0.5), 4)
        print(f"RERANK_THRESHOLD = {RERANK_THRESHOLD} TIDAK memisahkan kedua kelompok. "
              f"Set RERANK_THRESHOLD={suggested} di .env lalu restart service app.")


if __name__ == "__main__":
    main()
