"""Memuat PDF UU/PP, membersihkan noise OCR, metadata enrichment, dan parent/child splitting."""
import json
import os
import re

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from legal_rag.config import (
    CHILD_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, DOC_CATALOG, DOC_DIR, GDRIVE_FOLDER_URL,
    PARENT_CHUNK_OVERLAP, PARENT_CHUNK_SIZE, PARENTS_PATH,
)

HEADER_RE = re.compile(r"(NDONES|PRES.?DEN|FRES.?DEN|REPU|PERATURAN\.GO\.ID|^SALINAN$|^SK NO|^\d{4}, NO\.\s?\d+$)")
PAGE_NO_RE = re.compile(r"^(-\s*.{1,5}\s*-|-?\s*\d{1,4}\s*-?)$")
CONTINUATION_RE = re.compile(r"(\.\s?\.\s?\.|…)$")
OCR_FIXES = [
    (re.compile(r"\((\d)1(?=[\s.;,])"), r"(\1)"),          # "(21 Upah" -> "(2) Upah"
    (re.compile(r"\b(dan|kabupaten)l(atau|kota)\b"), r"\1/\2"),
    (re.compile(r"(?<=\d)O|O(?=\d)"), "0"),                 # 2O21 -> 2021
]
PASAL_LINE_RE = re.compile(r"^Pasal (\d+[A-Z]?)$")
BAB_LINE_RE = re.compile(r"^BAB ([IVXLC]+)$")
STRUCT_START = re.compile(r"^(Pasal \d+[A-Z]?$|BAB [IVXLC]+$|Bagian |Paragraf |\(\d+\)|[a-z]\.\s|\d+\.\s)")


def missing_docs(folder=DOC_DIR):
    return [f for f in DOC_CATALOG if not os.path.exists(os.path.join(folder, f))]


def download_docs(folder=DOC_DIR):
    if missing_docs(folder):
        import gdown
        gdown.download_folder(url=GDRIVE_FOLDER_URL, output=folder, quiet=False)
    assert not missing_docs(folder), f"PDF tidak ditemukan di folder Google Drive: {missing_docs(folder)}"


def clean_page(text):
    """Bersihkan satu halaman PDF. Return (teks_bersih, pasal_yang_dimulai, bab_yang_dimulai)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    keep = []
    for i, line in enumerate(lines):
        if len(line) < 35 and HEADER_RE.search(line.upper()):
            continue                                   # kop + footer
        if i < 5 and PAGE_NO_RE.match(line):
            continue                                   # nomor halaman
        keep.append(line)
    if keep and CONTINUATION_RE.search(keep[-1]) and len(keep[-1]) < 40:
        keep = keep[:-1]                               # penanda sambungan halaman
    text = "\n".join(keep)
    for pattern, replacement in OCR_FIXES:
        text = pattern.sub(replacement, text)
    lines = text.split("\n")
    pasal = [m.group(1) for line in lines if (m := PASAL_LINE_RE.match(line))]
    bab = [m.group(1) for line in lines if (m := BAB_LINE_RE.match(line))]
    # gabungkan baris hasil word-wrap PDF; baris berpenanda struktur (Pasal, ayat, huruf) tetap baris baru
    merged = []
    for line in lines:
        if merged and not STRUCT_START.match(line) and not PASAL_LINE_RE.match(merged[-1]):
            merged[-1] = merged[-1] + " " + line
        else:
            merged.append(line)
    return re.sub(r"[ \t]+", " ", "\n".join(merged)).strip(), pasal, bab


def load_pages(folder=DOC_DIR):
    """Satu Document per halaman berisi teks, diperkaya metadata peraturan + pasal/bab yang berlaku."""
    pages, load_stats = [], []
    for file_name, doc_meta in DOC_CATALOG.items():
        raw_pages = PyMuPDFLoader(os.path.join(folder, file_name)).load()
        current_pasal, current_bab, kept = "-", "-", 0
        for raw in raw_pages:
            text, pasal_started, bab_started = clean_page(raw.page_content)
            # pasal dari halaman sebelumnya masih berlaku bila halaman tidak diawali "Pasal N"
            carried = [current_pasal] if current_pasal != "-" and not text.startswith("Pasal") else []
            pasal_on_page = list(dict.fromkeys(carried + pasal_started))
            current_pasal = pasal_started[-1] if pasal_started else current_pasal
            current_bab = bab_started[-1] if bab_started else current_bab
            if len(text) < 100:
                continue
            kept += 1
            pages.append(Document(page_content=text, metadata={
                **doc_meta,
                "source": file_name,
                "page": raw.metadata["page"] + 1,
                "total_pages": raw.metadata["total_pages"],
                "pasal": ", ".join(pasal_on_page) or "-",
                "bab": current_bab,
            }))
        load_stats.append({"dokumen": doc_meta["singkatan"], "halaman PDF": len(raw_pages), "halaman berisi teks": kept})
    return pages, load_stats


def make_parent_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP, separators=["\nPasal ", "\n", ". ", " ", ""]
    )


def make_child_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP, separators=["\n", ". ", " ", ""]
    )


def split_parents(pages):
    parent_docs = make_parent_splitter().split_documents(pages)
    for i, doc in enumerate(parent_docs):
        doc.metadata["parent_id"] = f"{doc.metadata['peraturan_id']}-hal{doc.metadata['page']}-{i}"
        doc.metadata["char_len"] = len(doc.page_content)
    return parent_docs


def save_parents(parent_docs, path=PARENTS_PATH):
    """Parent chunks disimpan ke JSONL agar docstore + BM25 bisa dibangun ulang tanpa memproses PDF lagi."""
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for doc in parent_docs:
            f.write(json.dumps({"page_content": doc.page_content, "metadata": doc.metadata}, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def load_parents(path=PARENTS_PATH):
    with open(path, encoding="utf-8") as f:
        return [Document(**json.loads(line)) for line in f if line.strip()]
