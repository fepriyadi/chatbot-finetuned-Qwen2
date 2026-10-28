"""Konfigurasi bersama: katalog dokumen, ukuran chunk, model, prompt, dan endpoint vLLM.

Nilai yang perlu diganti per lingkungan dibaca dari environment variable (lihat .env.example).
"""
import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")   # matikan telemetry chromadb (sebelum chromadb di-import)

import torch

# --- Lokasi data (dibuat oleh `python -m legal_rag.ingest`) ---
DATA_DIR = os.environ.get("DATA_DIR", "data")
DOC_DIR = os.path.join(DATA_DIR, "dokumen_uu_pp")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma")
PARENTS_PATH = os.path.join(DATA_DIR, "parents.jsonl")   # ditulis terakhir -> penanda ingest selesai

# Keempat PDF diunduh langsung dari folder Google Drive (akses: "Anyone with the link").
GDRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1wGyyLRvuuJrroWb2TKmcBe_tY2c-huqY?usp=drive_link"

DOC_CATALOG = {
    "UU Nomor 6 Tahun 2023.pdf": dict(
        peraturan_id="UU-6-2023", jenis="UU", nomor=6, tahun=2023, singkatan="UU 6/2023",
        judul="Penetapan Perppu Nomor 2 Tahun 2022 tentang Cipta Kerja menjadi Undang-Undang",
        topik="cipta kerja"),
    "PP Nomor 35 Tahun 2021.pdf": dict(
        peraturan_id="PP-35-2021", jenis="PP", nomor=35, tahun=2021, singkatan="PP 35/2021",
        judul="Perjanjian Kerja Waktu Tertentu, Alih Daya, Waktu Kerja dan Waktu Istirahat, dan Pemutusan Hubungan Kerja",
        topik="ketenagakerjaan"),
    "PP Nomor 51 Tahun 2023.pdf": dict(
        peraturan_id="PP-51-2023", jenis="PP", nomor=51, tahun=2023, singkatan="PP 51/2023",
        judul="Perubahan atas Peraturan Pemerintah Nomor 36 Tahun 2021 tentang Pengupahan",
        topik="pengupahan"),
    "PP Nomor 5 Tahun 2021.pdf": dict(
        peraturan_id="PP-5-2021", jenis="PP", nomor=5, tahun=2021, singkatan="PP 5/2021",
        judul="Penyelenggaraan Perizinan Berusaha Berbasis Risiko",
        topik="perizinan berusaha"),
}

# --- Chunking ---
PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP = 2000, 200
CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP = 400, 80

# --- Retrieval ---
EMBEDDING_MODEL = "BAAI/bge-m3"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
COLLECTION = "uu_pp_child_chunks"
K_CHILD = 20                    # child chunk teratas -> beberapa parent unik
BM25_K = 10
ENSEMBLE_WEIGHTS = [0.4, 0.6]   # [BM25, semantik]
N_HYPOTHESES = 2
N_CANDIDATES = 10               # kandidat dari ensemble (>= 5) yang dinilai ulang reranker
TOP_K = 3
RERANK_THRESHOLD = float(os.environ.get("RERANK_THRESHOLD", "0.05"))   # kalibrasi ulang: python -m legal_rag.calibrate

DEVICE = os.environ.get("DEVICE") or (
    "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# --- Generator (model GRPO Kriteria 1, di-serve oleh vLLM) ---
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
LLM_MODEL = os.environ.get("LLM_MODEL", "Fepriyadi/Qwen2.5-3B-Instruct-Alpaca-ID-GRPO")
HF_TOKEN = os.environ.get("HF_TOKEN") or None

# --- Prompt (sama persis dengan notebook RAG / GRPO) ---
THINK_SYSTEM_PROMPT = (
    "Anda adalah asisten AI berbahasa Indonesia. Sebelum menjawab, tuliskan proses berpikir Anda "
    "langkah demi langkah di dalam tag <think> ... </think>. Setelah tag </think>, tuliskan jawaban akhir "
    "yang jelas dan ringkas dalam Bahasa Indonesia.\n"
    "Format:\n<think>\nproses berpikir\n</think>\njawaban akhir"
)

RAG_SYSTEM_PROMPT = THINK_SYSTEM_PROMPT + (
    "\nAnda adalah asisten hukum untuk Legal Team perusahaan di Indonesia. Jawab berdasarkan konteks peraturan "
    "yang diberikan dan cantumkan nomor sumber seperti [1] atau [2] di jawaban akhir."
)

# Kutip dulu, baru simpulkan. Tanpa langkah ini model memparafrasekan pasal dari ingatan dan pernah membalik syarat
# PP 35/2021 Pasal 26 ayat (1) "paling lama 4 jam" (maksimal) menjadi "4 jam atau lebih", sehingga lembur 3 jam
# dijawab "tidak berhak" padahal Pasal 27 ayat (1) mewajibkan Upah Kerja Lembur. Jawaban akhirnya juga bertentangan
# dengan <think> ("tidak dapat menentukan" lalu "tidak berhak"), maka "tidak ditemukan" hanya untuk kasus tanpa kutipan.
RAG_PROMPT = """Gunakan HANYA konteks berikut untuk menjawab pertanyaan.
Setiap potongan konteks diberi nomor sumber [n].

Di dalam <think>, kerjakan tiga langkah ini:
1. Kutip kata demi kata kalimat pasal yang relevan beserta nomor sumbernya, misalnya: [2] Pasal 26 ayat (1): "...".
2. Cocokkan fakta pada pertanyaan (angka, jabatan, waktu) dengan syarat di kutipan. Perhatikan kata batas:
   "paling lama" berarti maksimal, "paling sedikit" berarti minimal.
3. Tulis kesimpulan.

Setelah </think>, tulis jawaban akhir yang sama dengan kesimpulan di <think>, lalu sebutkan nama peraturan,
pasal, dan nomor sumber [n] yang Anda pakai.
Hanya jika langkah 1 tidak menemukan kutipan yang relevan, katakan bahwa informasi tersebut tidak ditemukan di dokumen.

Konteks:
{context}

Pertanyaan: {question}"""

HYDE_SYSTEM_PROMPT = (
    "Anda adalah ahli hukum ketenagakerjaan Indonesia. Tulis satu paragraf singkat (3-5 kalimat) bergaya pasal "
    "peraturan perundang-undangan Indonesia yang kemungkinan besar menjawab pertanyaan pengguna. "
    "Langsung tulis paragrafnya."
)

THINK_PREFIX = "<think>\n"
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
