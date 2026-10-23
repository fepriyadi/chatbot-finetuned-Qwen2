"""Pipeline RAG lengkap: metadata filter -> HyDE -> Ensemble (BM25 + semantik) -> Reranker Top-3 ->
threshold/DuckDuckGo -> LLM fine-tuned (vLLM) dengan {context} + {question}."""
from langchain_classic.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain_core.documents import Document
from langchain_core.stores import InMemoryStore
from sentence_transformers import CrossEncoder

from legal_rag.config import (
    DEVICE, ENSEMBLE_WEIGHTS, HYDE_SYSTEM_PROMPT, K_CHILD, N_CANDIDATES, N_HYPOTHESES, RAG_PROMPT,
    RAG_SYSTEM_PROMPT, RERANK_THRESHOLD, RERANKER_MODEL, THINK_PREFIX, TOP_K,
)
from legal_rag.llm import VLLMClient, extract_final_answer, extract_thinking
from legal_rag.preprocess import load_parents, make_child_splitter
from legal_rag.retrieval import (
    FilteredBM25Retriever, HyDEParentRetriever, PARENT_ID_KEY, detect_metadata_filter, load_embeddings,
    open_vectorstore, to_chroma_filter,
)


def duckduckgo_search(query, max_results=5):
    """Cari di internet via DuckDuckGo; hasil dibungkus Document agar formatnya sama dengan dokumen lokal."""
    from ddgs import DDGS
    try:
        results = DDGS().text(query, region="id-id", safesearch="moderate", max_results=max_results)
    except Exception as error:
        print("DuckDuckGo gagal:", error)
        results = []
    return [
        Document(
            page_content=f"{r.get('title', '')}\n{r.get('body', '')}",
            metadata={"source_type": "web", "title": r.get("title", ""), "url": r.get("href", "")},
        )
        for r in results
    ]


def format_context(docs):
    blocks = []
    for i, doc in enumerate(docs, start=1):
        m = doc.metadata
        if m.get("source_type") == "web":
            header = f"[{i}] {m['title']} ({m['url']})"
        else:
            header = f"[{i}] {m['singkatan']} tentang {m['judul']} — Pasal {m['pasal']}, halaman {m['page']}"
        blocks.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(blocks)


def citation_table(docs):
    """Tabel sitasi Markdown dari dokumen yang benar-benar dikirim ke LLM."""
    is_web = bool(docs) and docs[0].metadata.get("source_type") == "web"
    rows = [f"| No | Dokumen | Pasal | Halaman | {'URL' if is_web else 'Skor reranker'} |", "|---|---|---|---|---|"]
    for i, doc in enumerate(docs, start=1):
        m = doc.metadata
        if m.get("source_type") == "web":
            rows.append(f"| [{i}] | {m['title']} | - | - | [{m['url']}]({m['url']}) |")
        else:
            rows.append(f"| [{i}] | {m['singkatan']} — {m['judul']} | {m['pasal']} | {m['page']} | {m.get('rerank_score', '-')} |")
    return "\n".join(rows)


class LegalRAG:
    """Semua resource berat dimuat sekali (di Streamlit dibungkus st.cache_resource)."""

    def __init__(self, llm=None):
        self.parent_docs = load_parents()
        self.embedding_model = load_embeddings()
        self.vectorstore = open_vectorstore(self.embedding_model)
        self.docstore = InMemoryStore()
        self.docstore.mset([(d.metadata["parent_id"], d) for d in self.parent_docs])
        self.child_splitter = make_child_splitter()
        self.reranker = CrossEncoder(RERANKER_MODEL, max_length=512, device=DEVICE)
        self.llm = llm or VLLMClient()
        self.bm25_cache = {}

    def generate_hypotheses(self, question, n=N_HYPOTHESES):
        messages = [{"role": "system", "content": HYDE_SYSTEM_PROMPT}, {"role": "user", "content": question}]
        outputs = self.llm.generate_chat(messages, max_new_tokens=220, temperature=0.7, num_return_sequences=n)
        # model GRPO terbiasa menulis <think>; ambil jawaban akhir, atau isi penalarannya jika belum selesai
        return [extract_final_answer(o) or extract_thinking(o) or o for o in outputs]

    def build_hybrid_retriever(self, metadata_filter=None, use_hyde=True):
        """Ensemble BM25 + semantik (HyDE atau query biasa), keduanya menghormati metadata filter."""
        sparse = FilteredBM25Retriever(documents=self.parent_docs, metadata_filter=metadata_filter, cache=self.bm25_cache)
        if use_hyde:
            dense = HyDEParentRetriever(
                vectorstore=self.vectorstore, docstore=self.docstore, embeddings=self.embedding_model,
                hypothesis_fn=self.generate_hypotheses, id_key=PARENT_ID_KEY, k_child=K_CHILD,
                metadata_filter=metadata_filter,
            )
        else:
            dense = ParentDocumentRetriever(
                vectorstore=self.vectorstore, docstore=self.docstore, child_splitter=self.child_splitter,
                id_key=PARENT_ID_KEY, search_kwargs={"k": K_CHILD, "filter": to_chroma_filter(metadata_filter)},
            )
        return EnsembleRetriever(retrievers=[sparse, dense], weights=ENSEMBLE_WEIGHTS, id_key="parent_id"), dense

    def rerank(self, question, docs, top_k=TOP_K):
        """Top-k (dokumen, skor). Dokumen disalin agar rerank_score tidak menempel di docstore bersama."""
        if not docs:
            return []
        scores = self.reranker.predict([(question, d.page_content) for d in docs], batch_size=8, show_progress_bar=False)
        ranked = sorted(zip(docs, map(float, scores)), key=lambda pair: pair[1], reverse=True)[:top_k]
        return [
            (Document(page_content=doc.page_content, metadata={**doc.metadata, "rerank_score": round(score, 4)}), score)
            for doc, score in ranked
        ]

    def select_context(self, question, candidates, threshold=RERANK_THRESHOLD):
        """Rerank -> Top-3. Jika skor Top-1 < threshold, abaikan dokumen lokal dan pakai DuckDuckGo."""
        ranked = self.rerank(question, candidates)
        top1_score = ranked[0][1] if ranked else 0.0
        if top1_score >= threshold:
            return "dokumen lokal", [d for d, _ in ranked], top1_score
        return "internet (DuckDuckGo)", duckduckgo_search(question), top1_score

    def answer(self, question, metadata_filter=None, use_hyde=True):
        metadata_filter = metadata_filter or detect_metadata_filter(question)
        retriever, dense = self.build_hybrid_retriever(metadata_filter, use_hyde=use_hyde)
        candidates = retriever.invoke(question)[:N_CANDIDATES]
        route, context_docs, top1_score = self.select_context(question, candidates)

        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": RAG_PROMPT.format(context=format_context(context_docs), question=question)},
        ]
        raw_output = self.llm.generate_chat(messages, max_new_tokens=640, assistant_prefix=THINK_PREFIX)[0]
        return {
            "question": question,
            "metadata_filter": metadata_filter,
            "hypotheses": getattr(dense, "last_hypotheses", []),
            "n_candidates": len(candidates),
            "route": route,
            "top1_score": top1_score,
            "context_docs": context_docs,
            "raw_output": raw_output,
            "reasoning": extract_thinking(raw_output),
            "answer": extract_final_answer(raw_output) or raw_output,
        }
