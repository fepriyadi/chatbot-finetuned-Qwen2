"""Bangun indeks sekali: PDF -> parent/child chunks -> ChromaDB persisten + parents.jsonl.

    python -m legal_rag.ingest                  # bangun ulang dari nol
    python -m legal_rag.ingest --skip-if-exists # dipakai entrypoint Docker
"""
import argparse
import os
import time
from collections import Counter

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.stores import InMemoryStore

from legal_rag.config import DATA_DIR, DOC_DIR, K_CHILD, PARENTS_PATH
from legal_rag.preprocess import download_docs, load_pages, make_child_splitter, save_parents, split_parents
from legal_rag.retrieval import load_embeddings, open_vectorstore

BATCH = 256


def build_index():
    os.makedirs(DATA_DIR, exist_ok=True)
    download_docs(DOC_DIR)

    pages, load_stats = load_pages(DOC_DIR)
    for row in load_stats:
        print(f"{row['dokumen']:12s} halaman PDF {row['halaman PDF']:4d} | berisi teks {row['halaman berisi teks']:4d}")
    parent_docs = split_parents(pages)
    print(f"Total halaman: {len(pages)} | parent chunks: {len(parent_docs)}")
    print(dict(Counter(d.metadata["singkatan"] for d in parent_docs)))

    vectorstore = open_vectorstore(load_embeddings(), reset=True)
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=InMemoryStore(),          # docstore runtime dibangun ulang dari parents.jsonl
        child_splitter=make_child_splitter(),
        search_kwargs={"k": K_CHILD},
    )
    start = time.time()
    parent_ids = [d.metadata["parent_id"] for d in parent_docs]
    for i in range(0, len(parent_docs), BATCH):
        parent_retriever.add_documents(parent_docs[i:i + BATCH], ids=parent_ids[i:i + BATCH])
    print(f"Child vectors di ChromaDB: {vectorstore._collection.count()} | waktu: {time.time() - start:.0f} detik")

    save_parents(parent_docs)              # terakhir: keberadaan file ini = indeks lengkap
    print("Parent chunks disimpan ke", PARENTS_PATH)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skip-if-exists", action="store_true", help="lewati bila indeks sudah ada")
    args = parser.parse_args()
    if args.skip_if_exists and os.path.exists(PARENTS_PATH):
        print(f"Indeks sudah ada ({PARENTS_PATH}), ingest dilewati.")
        return
    build_index()


if __name__ == "__main__":
    main()
