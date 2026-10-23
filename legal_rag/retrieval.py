"""Embedding, ChromaDB persisten, metadata filter, BM25, HyDE, dan ensemble retriever."""
import json
import re
from typing import Any, Optional

import chromadb
import torch
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from legal_rag.config import BM25_K, CHROMA_DIR, COLLECTION, DEVICE, DOC_CATALOG, EMBEDDING_MODEL

PARENT_ID_KEY = "doc_id"   # key metadata child -> parent (default ParentDocumentRetriever)


def load_embeddings(device=DEVICE):
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
    )


def open_vectorstore(embedding_model, reset=False):
    """ChromaDB lokal persisten (child chunks, cosine distance). reset=True -> koleksi dihapus dulu."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    if reset:
        try:
            client.delete_collection(COLLECTION)        # aman di-run ulang tanpa duplikasi vektor
        except Exception:
            pass
    return Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"},
    )


# --- Metadata filtering ---

def to_chroma_filter(metadata_filter):
    """{"jenis": "PP", "tahun": 2021} -> sintaks filter ChromaDB."""
    if not metadata_filter:
        return None
    clauses = [{key: value} for key, value in metadata_filter.items()]
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def matches_filter(metadata, metadata_filter):
    return not metadata_filter or all(metadata.get(k) == v for k, v in metadata_filter.items())


REFERENCE_RE = re.compile(
    r"\b(PP|UU|Peraturan Pemerintah|Undang[- ]Undang)\s*(?:No(?:mor|\.)?\s*)?(\d+)\s*(?:/|Tahun)\s*(\d{4})",
    re.IGNORECASE,
)


def detect_metadata_filter(question):
    """Rujukan peraturan eksplisit di pertanyaan -> filter peraturan_id (hanya untuk 4 dokumen yang tersedia)."""
    known = {m["peraturan_id"] for m in DOC_CATALOG.values()}
    for jenis, nomor, tahun in REFERENCE_RE.findall(question):
        jenis = "PP" if jenis.upper().startswith("P") else "UU"
        peraturan_id = f"{jenis}-{int(nomor)}-{tahun}"
        if peraturan_id in known:
            return {"peraturan_id": peraturan_id}
    return None


# --- BM25 ---

STOPWORDS_ID = set("""yang dan di ke dari dalam untuk pada dengan atau ini itu adalah sebagaimana dimaksud ayat
huruf oleh tidak dapat akan bagi sebagai juga serta tersebut telah atas secara para saya apakah kemarin anda kami
bisa mau dapat""".split())


def tokenize_id(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS_ID and len(t) > 1]


class FilteredBM25Retriever(BaseRetriever):
    """BM25 atas parent chunks yang lolos metadata filter (indeks per filter di-cache)."""

    documents: list
    k: int = BM25_K
    metadata_filter: Optional[dict] = None
    cache: Any = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        key = json.dumps(self.metadata_filter, sort_keys=True)
        if key not in self.cache:
            docs = [d for d in self.documents if matches_filter(d.metadata, self.metadata_filter)]
            self.cache[key] = BM25Retriever.from_documents(docs, k=self.k, preprocess_func=tokenize_id)
        return self.cache[key].invoke(query)


# --- HyDE ---

class HyDEParentRetriever(BaseRetriever):
    """Vector search dengan embedding gabungan (query + jawaban hipotetis) -> parent chunks."""

    vectorstore: Any
    docstore: Any
    embeddings: Any
    hypothesis_fn: Any
    id_key: str = PARENT_ID_KEY
    k_child: int = 20
    metadata_filter: Optional[dict] = None
    last_hypotheses: list = []

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        hypotheses = self.hypothesis_fn(query)
        self.last_hypotheses = hypotheses
        vectors = [self.embeddings.embed_query(query)] + self.embeddings.embed_documents(hypotheses)
        combined = torch.tensor(vectors).mean(dim=0)
        combined = torch.nn.functional.normalize(combined, dim=0).tolist()
        children = self.vectorstore.similarity_search_by_vector(
            combined, k=self.k_child, filter=to_chroma_filter(self.metadata_filter)
        )
        ordered_ids = list(dict.fromkeys(child.metadata[self.id_key] for child in children))
        return [doc for doc in self.docstore.mget(ordered_ids) if doc is not None]
