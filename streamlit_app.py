"""Interface Streamlit untuk Asisten Hukum Legal Team (pengganti `legal_assistant_loop` notebook)."""
import streamlit as st

from legal_rag.config import DOC_CATALOG, LLM_MODEL, RERANK_THRESHOLD, VLLM_BASE_URL
from legal_rag.pipeline import LegalRAG, citation_table

st.set_page_config(page_title="Asisten Hukum Legal Team", page_icon="⚖️", layout="wide")

AUTO_FILTER = "Otomatis (deteksi dari pertanyaan)"
FILTER_OPTIONS = {AUTO_FILTER: None, **{m["singkatan"]: {"peraturan_id": m["peraturan_id"]} for m in DOC_CATALOG.values()}}
EXAMPLES = [
    "Saya staf admin, kemarin lembur 3 jam untuk beresin laporan. Apakah saya berhak dapat uang lembur?",
    "Menurut PP 35/2021, berapa lama jangka waktu paling lama PKWT dan bagaimana perpanjangannya?",
    "Bagaimana formula penghitungan upah minimum?",
]


@st.cache_resource(show_spinner="Memuat embedding, reranker, dan indeks dokumen...")
def get_rag():
    return LegalRAG()


def render_result(result):
    route_note = (
        f"skor reranker Top-1 = `{result['top1_score']:.4f}` "
        f"{'≥' if result['route'] == 'dokumen lokal' else '<'} threshold `{RERANK_THRESHOLD}`"
    )
    st.caption(
        f"**Sumber konteks:** {result['route']} ({route_note}) · **Filter metadata:** `{result['metadata_filter']}` "
        f"· **Kandidat ensemble:** {result['n_candidates']}"
    )
    with st.expander("Proses berpikir model", expanded=False):
        st.code(f"<think>\n{result['reasoning']}\n</think>", language="text")   # tag <think> tetap terlihat
    st.markdown(result["answer"])
    st.markdown("**Sitasi**")
    st.markdown(citation_table(result["context_docs"]))
    if result["hypotheses"]:
        with st.expander("Jawaban hipotetis HyDE"):
            for i, hypothesis in enumerate(result["hypotheses"], start=1):
                st.markdown(f"**[{i}]** {hypothesis}")
    with st.expander("Output mentah model"):
        st.code(result["raw_output"], language="text")


rag = get_rag()

with st.sidebar:
    st.header("Pengaturan")
    use_hyde = st.toggle("HyDE (2 jawaban hipotetis)", value=True,
                         help="Membantu pertanyaan bahasa percakapan; menambah 1 panggilan LLM.")
    filter_label = st.selectbox("Filter peraturan", list(FILTER_OPTIONS))
    st.divider()
    if rag.llm.is_ready():
        st.success(f"vLLM siap: `{LLM_MODEL}`")
    else:
        st.error(f"vLLM belum siap di `{VLLM_BASE_URL}`")
    if st.button("Hapus riwayat chat"):
        st.session_state.history = []
        st.rerun()

st.title("⚖️ Asisten Hukum Legal Team")
st.markdown("Tanya jawab berdasarkan **UU 6/2023, PP 35/2021, PP 51/2023, dan PP 5/2021**. "
            "Pertanyaan di luar dokumen dijawab dari internet (DuckDuckGo).")

if "history" not in st.session_state:
    st.session_state.history = []

for item in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(item["question"])
    with st.chat_message("assistant"):
        render_result(item)

question = st.chat_input("Ketik pertanyaan Anda...") or st.session_state.pop("pending_question", None)

if not st.session_state.history and not question:
    st.markdown("**Contoh pertanyaan:**")
    for example in EXAMPLES:
        if st.button(example):
            st.session_state.pending_question = example
            st.rerun()

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Mencari pasal relevan dan menyusun jawaban..."):
            try:
                result = rag.answer(question, metadata_filter=FILTER_OPTIONS[filter_label], use_hyde=use_hyde)
            except Exception as error:
                st.error(f"Gagal menghasilkan jawaban: {error}")
                st.stop()
        render_result(result)
    st.session_state.history.append(result)
