"""Bangun dataset SFT RAG: contoh kerja "kutip -> bandingkan -> simpulkan" dari keempat PDF sendiri.

    python -m legal_rag.build_sft_data --dry-run            # tahap 1 saja: pasal kandidat, tanpa model / indeks
    python -m legal_rag.build_sft_data --stage teacher      # tahap 1-2: butuh vLLM teacher (TEACHER_BASE_URL)
    python -m legal_rag.build_sft_data --stage build        # tahap 3-5: butuh indeks ingest + vLLM model GRPO (HyDE)
    python -m legal_rag.build_sft_data                      # semua tahap, bila kedua server vLLM jalan bersamaan
    python -m legal_rag.build_sft_data --stage build --no-hyde   # retrieval tanpa HyDE bila vLLM GRPO tidak ada

Teacher = Qwen/Qwen2.5-32B-Instruct-AWQ (open-source) di-serve vLLM. Di Kaggle 2x T4 teacher 32B dan model GRPO
tidak muat bersamaan, maka tahap dipisah: jalankan teacher, matikan, lalu jalankan vLLM GRPO untuk --stage build.

    vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ --port 8001 --tensor-parallel-size 2 --quantization awq \
        --dtype half --max-model-len 8192 --gpu-memory-utilization 0.92

Kenapa SFT: model GRPO bisa mengutip "paling lama 4 jam" tetapi gagal menerapkannya ("3 jam tidak termasuk
dalam batas"). GRPO tidak pernah melihat prompt RAG (MAX_PROMPT_TOKENS = 256), jadi perilaku itu harus
dicontohkan, bukan diberi reward.

Tahap:
1. Pilih parent chunk yang memuat batas angka ("paling lama 5 (lima) tahun"). Chunk yang menyebut "lembur"
   dibuang: PP 35/2021 Pasal 26-27 adalah test case dan tidak boleh bocor ke data latih.
2. Teacher mengekstrak kalimat hak/kewajiban + kalimat batas secara verbatim, lalu menulis pertanyaan
   dengan label yang sudah diketahui: dalam batas / melewati batas / pengecualian.
3. Pertanyaan dijalankan lewat retrieval asli (filter -> HyDE -> ensemble -> rerank Top-3), sehingga konteks
   latih sama berisiknya dengan konteks saat inferensi.
4. Target <think> dirakit oleh kode dari hasil ekstraksi, bukan ditulis bebas oleh teacher.
5. Verifikasi: kutipan verbatim ada di konteks, [n] menunjuk dokumen yang memuatnya, pasal cocok dengan metadata,
   label teacher sama dengan label yang dihitung ulang dari angka. Gagal satu -> masuk rejects.jsonl.

Output di data/rag_sft/: teacher_cache.jsonl, train.jsonl, eval.jsonl, rejects.jsonl.
"""
import argparse
import json
import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Literal

from pydantic import BaseModel

from legal_rag.config import (
    DATA_DIR, N_CANDIDATES, RAG_PROMPT, RAG_SYSTEM_PROMPT, RERANK_THRESHOLD, TEACHER_BASE_URL, TEACHER_MODEL,
)
from legal_rag.preprocess import load_parents

OUT_DIR = os.path.join(DATA_DIR, "rag_sft")
SEED = 3407
MAX_SEQ_LENGTH = 4096         # sama dengan max_seq_length model di notebook RAG
EVAL_RULE_RATIO = 0.1         # dipisah per aturan (bukan per pertanyaan) agar varian satu pasal tidak bocor ke eval
NOT_FOUND_RATIO = 0.1         # porsi contoh "tidak ditemukan" terhadap contoh positif

LIMIT_RE = re.compile(r"paling (?:lama|sedikit|singkat|lambat|banyak|tinggi|rendah|cepat)\s+(?:Rp\s?)?\d", re.I)
HOLDOUT_RE = re.compile(r"lembur", re.I)
NOT_FOUND_ANSWER = "Informasi tersebut tidak ditemukan di dokumen yang tersedia."
GENERIC_WORDS = {"jangka", "waktu", "jumlah", "paling", "hari", "kerja", "bulan", "tahun", "besaran", "batas", "lama"}


# --- Skema keluaran teacher (structured outputs vLLM: JSON dipaksa mengikuti skema) ---

class Kutipan(BaseModel):
    pasal: str     # "8"; kosong jika tidak ada
    ayat: str      # "(1)"; kosong jika pasal tanpa ayat
    kalimat: str   # salinan verbatim dari teks; kosong jika tidak ada


class Pertanyaan(BaseModel):
    varian: Literal["dalam_batas", "melewati_batas", "pengecualian"]
    pertanyaan: str
    fakta: str
    nilai_fakta: float
    label: Literal["Ya", "Tidak"]
    jawaban: str


class Aturan(BaseModel):
    layak: bool
    alasan: str
    besaran: str
    satuan: str
    nilai_batas: float
    arah: Literal["maksimal", "minimal"]
    hak: Kutipan
    batas: Kutipan
    pengecualian: Kutipan
    pertanyaan: List[Pertanyaan]


TEACHER_SYSTEM = """Anda membantu membuat data latih untuk asisten hukum ketenagakerjaan dan perizinan Indonesia.
Anda menerima satu potongan teks peraturan (hasil OCR, bisa ada salah ketik). Kerjakan:

1. Pilih SATU ketentuan yang memuat batas angka ("paling lama", "paling sedikit", "paling lambat", "paling banyak",
   dst.) yang wajar ditanyakan oleh pekerja, HR, atau tim legal perusahaan dalam bentuk pertanyaan ya/tidak.
   Jika tidak ada yang wajar (misalnya hanya batas internal birokrasi pemerintah), isi layak=false, beri alasan,
   dan isi field lain seadanya.
2. Salin kalimatnya PERSIS seperti di teks, termasuk salah ketik OCR, tanpa memperbaiki atau meringkas:
   - batas: kalimat yang memuat angka batas.
   - hak: kalimat yang memberi hak atau kewajiban terkait (mis. "Pengusaha wajib ..."), jika berbeda dari kalimat
     batas dan ada di teks. Jika tidak ada, isi semua field hak dengan string kosong.
   - pengecualian: kalimat pengecualian terkait (mis. "tidak berlaku bagi ...", "kecuali ..."), jika ada di teks.
     Jika tidak ada, isi semua field pengecualian dengan string kosong.
   Isi pasal dengan nomor pasal saja ("8") dan ayat dengan bentuk "(1)" atau string kosong.
3. besaran = apa yang dibatasi (mis. "jangka waktu PKWT"), satuan = satuan angkanya (mis. "tahun", "hari kerja"),
   nilai_batas = angkanya, arah = "maksimal" untuk paling lama/lambat/banyak/tinggi, "minimal" untuk
   paling sedikit/singkat/rendah/cepat.
4. Tulis pertanyaan dalam bahasa percakapan orang pertama, seperti "Saya staf admin, kontrak saya 3 tahun.
   Apakah ini masih sesuai aturan?". Jangan sebut nama peraturan atau nomor pasal. Pertanyaan harus menyebut angka
   fakta secara eksplisit dan dibingkai sehingga "Ya" berarti sah / sesuai aturan / berhak.
   - dalam_batas: nilai_fakta memenuhi batas (<= untuk maksimal, >= untuk minimal), label "Ya".
   - melewati_batas: nilai_fakta tidak memenuhi batas, label "Tidak".
   - pengecualian: hanya jika kalimat pengecualian ada. Fakta memicu pengecualian sehingga label berlawanan
     dengan hasil perbandingan angka saja.
   Buat satu pertanyaan per varian yang mungkin. Variasikan peran penanya (pekerja, HR, pemilik usaha) dan
   gunakan angka yang tidak sama dengan nilai_batas.
5. fakta = ringkasan fakta pertanyaan dalam beberapa kata (mis. "kontrak 3 tahun"). jawaban = 1-2 kalimat yang
   diawali "Ya," atau "Tidak," sesuai label, menyebut alasan berdasarkan kalimat yang dikutip.

Jawab hanya dengan satu objek JSON sesuai skema."""


# --- Tahap 1: pasal kandidat ---

def select_parents(parent_docs):
    """Parent chunk dengan batas angka. Chunk tentang lembur ditahan sebagai test case."""
    candidates, held_out = [], 0
    for doc in parent_docs:
        if not LIMIT_RE.search(doc.page_content):
            continue
        if HOLDOUT_RE.search(doc.page_content):
            held_out += 1
            continue
        candidates.append(doc)
    return candidates, held_out


# --- Tahap 2: teacher ---

def inline_refs(schema):
    """Skema Pydantic memakai $defs/$ref; di-inline agar aman untuk backend structured outputs vLLM."""
    defs = schema.pop("$defs", {})

    def resolve(node):
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs[node["$ref"].rsplit("/", 1)[1]])
            return {key: resolve(value) for key, value in node.items()}
        if isinstance(node, list):
            return [resolve(value) for value in node]
        return node

    return resolve(schema)


TEACHER_SCHEMA = inline_refs(Aturan.model_json_schema())


def ask_teacher(client, doc, model):
    """Satu panggilan teacher per parent chunk lewat /v1/chat/completions vLLM."""
    m = doc.metadata
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TEACHER_SYSTEM},
            {"role": "user",
             "content": f"{m['singkatan']} tentang {m['judul']}, halaman {m['page']}\n\n{doc.page_content}"},
        ],
        max_tokens=2048,
        temperature=0.2,   # rendah: kutipan harus verbatim; variasi pertanyaan cukup dari isi tiap pasal
        response_format={"type": "json_schema", "json_schema": {"name": "aturan", "schema": TEACHER_SCHEMA}},
    )
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise ValueError("keluaran terpotong max_tokens")
    return Aturan.model_validate_json(choice.message.content).model_dump()


def load_teacher_cache(cache_path):
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                cache[row["parent_id"]] = row["aturan"]
    return cache


def run_teacher(candidates, cache_path, model, workers):
    """Hasil teacher di-cache per parent_id agar run ulang hanya mengulang panggilan yang gagal."""
    from openai import OpenAI, OpenAIError
    from pydantic import ValidationError

    cache = load_teacher_cache(cache_path)
    todo = [d for d in candidates if d.metadata["parent_id"] not in cache]
    print(f"Teacher: {len(cache)} dari cache, {len(todo)} panggilan baru ({model} @ {TEACHER_BASE_URL})")

    client = OpenAI(base_url=TEACHER_BASE_URL, api_key="EMPTY", timeout=600)
    with open(cache_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(ask_teacher, client, doc, model): doc for doc in todo}
        for i, future in enumerate(as_completed(futures), start=1):
            parent_id = futures[future].metadata["parent_id"]
            try:
                aturan = future.result()
            except (OpenAIError, ValidationError, ValueError) as error:
                print(f"  gagal {parent_id}: {str(error)[:200]}")   # tidak di-cache -> dicoba lagi pada run berikutnya
                continue
            cache[parent_id] = aturan
            f.write(json.dumps({"parent_id": parent_id, "aturan": aturan}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 20 == 0:
                print(f"  {i}/{len(todo)}")
    return cache


# --- Verifikasi + perakitan target ---

def norm(text):
    return re.sub(r"\s+", " ", text).strip()


def fmt_number(x):
    return str(int(x)) if float(x).is_integer() else str(x).replace(".", ",")


def mentions_number(text, x):
    """Angka x disebut di teks: "1 (satu)", "Rp1.000.000", "0,5" atau "0.5"."""
    if float(x).is_integer():
        return re.search(rf"(?<![\d,]){int(x)}(?!\d|,\d)", text.replace(".", "")) is not None
    return any(re.search(rf"(?<![\d.,]){re.escape(v)}(?![\d])", text) for v in {fmt_number(x), str(x)})


def within_limit(nilai_fakta, nilai_batas, arah):
    return nilai_fakta <= nilai_batas if arah == "maksimal" else nilai_fakta >= nilai_batas


def check_rule(aturan, source_doc):
    """Masalah pada hasil ekstraksi (sebelum retrieval). List kosong = lolos."""
    problems = []
    if not aturan["layak"]:
        return [f"tidak layak: {aturan['alasan']}"]
    source = norm(source_doc.page_content)
    for name in ["batas", "hak", "pengecualian"]:
        kalimat = norm(aturan[name]["kalimat"])
        if kalimat and kalimat not in source:
            problems.append(f"kutipan {name} tidak verbatim")
    if not aturan["batas"]["kalimat"]:
        problems.append("kalimat batas kosong")
    elif not mentions_number(aturan["batas"]["kalimat"], aturan["nilai_batas"]):
        problems.append("nilai_batas tidak ada di kalimat batas")
    return problems


def check_question(aturan, q):
    """Label dihitung ulang dari angka; teacher hanya dipercaya untuk arah pengecualian."""
    inside = within_limit(q["nilai_fakta"], aturan["nilai_batas"], aturan["arah"])
    if not q["jawaban"].startswith(q["label"]):
        return "jawaban tidak diawali label"
    if q["varian"] == "dalam_batas" and not (inside and q["label"] == "Ya"):
        return "dalam_batas tidak cocok dengan angka"
    if q["varian"] == "melewati_batas" and not (not inside and q["label"] == "Tidak"):
        return "melewati_batas tidak cocok dengan angka"
    if q["varian"] == "pengecualian":
        if not aturan["pengecualian"]["kalimat"]:
            return "pengecualian tanpa kalimat pengecualian"
        if q["label"] == ("Ya" if inside else "Tidak"):
            return "pengecualian tidak membalik hasil perbandingan"
    if not mentions_number(q["pertanyaan"], q["nilai_fakta"]):
        return "angka fakta tidak disebut di pertanyaan"
    return None


def locate(kutipan, context_docs):
    """Nomor sumber [n] dokumen konteks yang memuat kutipan verbatim dan pasalnya. None jika tidak ada."""
    kalimat = norm(kutipan["kalimat"])
    for i, doc in enumerate(context_docs, start=1):
        pasal_on_doc = [p.strip() for p in doc.metadata.get("pasal", "").split(",")]
        if kalimat in norm(doc.page_content) and kutipan["pasal"] in pasal_on_doc:
            return i
    return None


def cite(kutipan, n):
    ayat = f" ayat {kutipan['ayat']}" if kutipan["ayat"] else ""
    return f"[{n}] Pasal {kutipan['pasal']}{ayat}"


def build_target(aturan, q, context_docs):
    """Target assistant. Return (teks, alasan_tolak)."""
    n_batas = locate(aturan["batas"], context_docs)
    if n_batas is None:
        return None, "kalimat batas tidak ada di konteks Top-3"
    lines = []
    if aturan["hak"]["kalimat"]:
        n_hak = locate(aturan["hak"], context_docs)
        if n_hak is None:
            return None, "kalimat hak tidak ada di konteks Top-3"
        lines.append(f'Hak/kewajiban: {cite(aturan["hak"], n_hak)}: "{norm(aturan["hak"]["kalimat"])}"')
    lines.append(f'Batas: {cite(aturan["batas"], n_batas)}: "{norm(aturan["batas"]["kalimat"])}"')
    lines.append(f"Fakta: {q['fakta']}")

    satuan, batas, fakta = aturan["satuan"], fmt_number(aturan["nilai_batas"]), fmt_number(q["nilai_fakta"])
    inside = within_limit(q["nilai_fakta"], aturan["nilai_batas"], aturan["arah"])
    if aturan["arah"] == "maksimal":
        op, status = ("≤", "masih di dalam batas maksimal") if inside else (">", "melewati batas maksimal")
    else:
        op, status = ("≥", "memenuhi batas minimal") if inside else ("<", "di bawah batas minimal")
    lines.append(f"Perbandingan: {fakta} {satuan} {op} {batas} {satuan} → {status}")

    if aturan["pengecualian"]["kalimat"]:
        n_kecuali = locate(aturan["pengecualian"], context_docs)
        if q["varian"] == "pengecualian":
            if n_kecuali is None:
                return None, "kalimat pengecualian tidak ada di konteks Top-3"
            lines.append(f'Pengecualian: {cite(aturan["pengecualian"], n_kecuali)}: '
                         f'"{norm(aturan["pengecualian"]["kalimat"])}" → berlaku pada fakta ini')
        elif n_kecuali is not None:
            lines.append(f"Pengecualian: {cite(aturan['pengecualian'], n_kecuali)} tidak berlaku, "
                         f"fakta di pertanyaan tidak memenuhinya")
        else:
            lines.append("Pengecualian: tidak ada di konteks")
    else:
        lines.append("Pengecualian: tidak ada")
    lines.append(f"Kesimpulan: {q['jawaban']}")

    m = context_docs[n_batas - 1].metadata
    pasal = cite(aturan["batas"], n_batas).split(" ", 1)[1]
    answer = f"Jawaban: {q['jawaban']} [{n_batas}] {m['singkatan']} {pasal}."
    return "<think>\n" + "\n".join(lines) + "\n</think>\n" + answer, None


def build_not_found_target(aturan):
    return ("<think>\n"
            f"Hak/kewajiban: tidak ada kalimat pasal di konteks yang mengatur {aturan['besaran']}.\n"
            "Kesimpulan: konteks tidak memuat aturan untuk pertanyaan ini.\n"
            f"</think>\nJawaban: {NOT_FOUND_ANSWER}")


def to_messages(question, context_docs, target, format_context):
    return [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": RAG_PROMPT.format(context=format_context(context_docs), question=question)},
        {"role": "assistant", "content": target},
    ]


# --- Tahap 3-5: retrieval, target, verifikasi ---

def build_examples(rules, rag, use_hyde):
    from legal_rag.pipeline import format_context
    from legal_rag.retrieval import detect_metadata_filter

    examples, rejects, retrieved = [], [], []
    for n_rule, (parent_id, aturan, source_doc) in enumerate(rules, start=1):
        for q in aturan["pertanyaan"]:
            row = {"parent_id": parent_id, "varian": q["varian"], "pertanyaan": q["pertanyaan"]}
            problem = check_question(aturan, q)
            if problem:
                rejects.append({**row, "alasan": problem})
                continue
            retriever, _ = rag.build_hybrid_retriever(detect_metadata_filter(q["pertanyaan"]), use_hyde=use_hyde)
            candidates = retriever.invoke(q["pertanyaan"])[:N_CANDIDATES]
            ranked = rag.rerank(q["pertanyaan"], candidates)
            if not ranked or ranked[0][1] < RERANK_THRESHOLD:
                rejects.append({**row, "alasan": "rute internet (skor Top-1 di bawah threshold)"})
                continue
            context_docs = [doc for doc, _ in ranked]
            target, problem = build_target(aturan, q, context_docs)
            if problem:
                rejects.append({**row, "alasan": problem})
                continue
            examples.append({**row, "label": q["label"],
                             "messages": to_messages(q["pertanyaan"], context_docs, target, format_context)})
            retrieved.append((parent_id, aturan, q, context_docs))
        if n_rule % 20 == 0:
            print(f"  retrieval {n_rule}/{len(rules)} aturan | contoh {len(examples)} | ditolak {len(rejects)}")

    # "tidak ditemukan": pertanyaan aturan A dengan konteks Top-3 milik pertanyaan dari peraturan lain, asalkan
    # konteks itu tidak memuat kalimat batas A maupun kata kunci khas besaran A ("PKWT", bukan "waktu").
    rng = random.Random(SEED)
    for parent_id, aturan, q, _ in rng.sample(retrieved, k=min(len(retrieved), int(len(retrieved) * NOT_FOUND_RATIO))):
        source_id = parent_id.split("-hal")[0]
        keywords = set(re.findall(r"[a-z]{4,}", aturan["besaran"].lower())) - GENERIC_WORDS
        for other_id, _, _, other_docs in rng.sample(retrieved, k=len(retrieved)):
            text = norm(" ".join(d.page_content for d in other_docs)).lower()
            if (other_id.split("-hal")[0] != source_id and norm(aturan["batas"]["kalimat"]).lower() not in text
                    and not any(w in text for w in keywords)):
                examples.append({"parent_id": parent_id, "varian": "tidak_ditemukan", "pertanyaan": q["pertanyaan"],
                                 "label": "Tidak ditemukan",
                                 "messages": to_messages(q["pertanyaan"], other_docs, build_not_found_target(aturan),
                                                         format_context)})
                break
    return examples, rejects


def split_by_rule(examples):
    rule_ids = sorted({e["parent_id"] for e in examples})
    random.Random(SEED).shuffle(rule_ids)
    eval_ids = set(rule_ids[:max(1, round(len(rule_ids) * EVAL_RULE_RATIO))])
    return [e for e in examples if e["parent_id"] not in eval_ids], [e for e in examples if e["parent_id"] in eval_ids]


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="hanya tahap 1 (pasal kandidat)")
    parser.add_argument("--stage", choices=["all", "teacher", "build"], default="all",
                        help="teacher = tahap 1-2 (vLLM teacher), build = tahap 3-5 dari teacher_cache.jsonl")
    parser.add_argument("--no-hyde", action="store_true", help="retrieval tanpa HyDE (tanpa vLLM model GRPO)")
    parser.add_argument("--teacher-model", default=TEACHER_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="batasi jumlah pasal kandidat (uji coba)")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    parent_docs = load_parents()
    candidates, held_out = select_parents(parent_docs)
    candidates = candidates[:args.limit] if args.limit else candidates
    print(f"Parent chunks: {len(parent_docs)} | kandidat batas angka: {len(candidates)} | "
          f"ditahan (lembur): {held_out}")
    print(dict(Counter(d.metadata["singkatan"] for d in candidates)))
    if args.dry_run:
        return

    cache_path = os.path.join(OUT_DIR, "teacher_cache.jsonl")
    if args.stage in ("all", "teacher"):
        cache = run_teacher(candidates, cache_path, args.teacher_model, args.workers)
    else:
        cache = load_teacher_cache(cache_path)

    rules, rejects = [], []
    for doc in candidates:
        parent_id = doc.metadata["parent_id"]
        aturan = cache.get(parent_id)
        problems = ["belum ada hasil teacher"] if aturan is None else check_rule(aturan, doc)
        if problems:
            rejects += [{"parent_id": parent_id, "varian": "-", "pertanyaan": "-", "alasan": p} for p in problems]
        else:
            rules.append((parent_id, aturan, doc))
    n_questions = sum(check_question(a, q) is None for _, a, _ in rules for q in a["pertanyaan"])
    print(f"Aturan lolos ekstraksi: {len(rules)} dari {len(candidates)} | pertanyaan lolos cek angka: {n_questions}")
    print("alasan tolak ekstraksi:", dict(Counter(r["alasan"].split(":")[0] for r in rejects).most_common(8)))
    if args.stage == "teacher" or not rules:
        return

    from legal_rag.pipeline import LegalRAG

    rag = LegalRAG()
    use_hyde = not args.no_hyde
    if use_hyde and not rag.llm.is_ready():
        raise SystemExit("vLLM model GRPO tidak dapat dihubungi untuk HyDE. Jalankan vLLM / tunnel, "
                         "atau pakai --no-hyde.")

    examples, question_rejects = build_examples(rules, rag, use_hyde)
    rejects += question_rejects

    # panjang token dihitung dengan chat template model yang akan dilatih
    kept = []
    for e in examples:
        text = rag.llm.tokenizer.apply_chat_template(e["messages"], tokenize=False)
        n_tokens = len(rag.llm.tokenizer(text, add_special_tokens=False)["input_ids"])
        if n_tokens > MAX_SEQ_LENGTH:
            rejects.append({k: e[k] for k in ["parent_id", "varian", "pertanyaan"]} | {"alasan": f"{n_tokens} token"})
        else:
            kept.append({**e, "n_tokens": n_tokens})

    train, eval_ = split_by_rule(kept)
    write_jsonl(train, os.path.join(OUT_DIR, "train.jsonl"))
    write_jsonl(eval_, os.path.join(OUT_DIR, "eval.jsonl"))
    write_jsonl(rejects, os.path.join(OUT_DIR, "rejects.jsonl"))
    print(f"\ntrain {len(train)} | eval {len(eval_)} | ditolak {len(rejects)}  -> {OUT_DIR}")
    print("varian train:", dict(Counter(e["varian"] for e in train)))
    print("alasan tolak:", dict(Counter(r["alasan"].split(":")[0] for r in rejects).most_common(8)))


if __name__ == "__main__":
    main()
