# chatbot-finetuned-Qwen2 — Legal Team RAG Assistant

Deployment of the RAG notebook (`PGABL_FepriyadiHarahap/RAG_submission_PGABL_FepriyadiH.ipynb`) as a
Streamlit app, with the fine-tuned GRPO model served by **vLLM**, packaged with **Docker Compose**.

```text
docker compose
    ├── vllm   (GPU)  Fepriyadi/Qwen2.5-3B-Instruct-Alpaca-ID-GRPO, OpenAI-compatible API :8000
    └── app    (GPU)  Streamlit :8501
          ├── bge-m3 embeddings + ChromaDB (persisted in ./data)
          ├── BM25 + parent docstore (./data/parents.jsonl)
          ├── bge-reranker-v2-m3  → Top-3, threshold → DuckDuckGo fallback
          └── HTTP → vllm  (HyDE hypotheses + final answer)
```

## Layout

```text
legal_rag/
    config.py       DOC_CATALOG, chunk sizes, models, prompts, env vars
    preprocess.py   PDF download, OCR-noise cleaning, metadata, parent/child split
    retrieval.py    embeddings, persistent Chroma, metadata filter, BM25, HyDE retriever
    llm.py          vLLM client (replaces the notebook's Unsloth generate_chat)
    pipeline.py     LegalRAG: ensemble → rerank → threshold/DuckDuckGo → LLM, citations
    ingest.py       build the index once
    calibrate.py    re-check RERANK_THRESHOLD on the deployment hardware
streamlit_app.py    chat UI
```

## What changed vs. the notebook

| Notebook | Here |
|---|---|
| Unsloth 4-bit model in-process, `llm.generate()` | vLLM server, fp16 (`merged_16bit` checkpoint served as-is) |
| `chromadb.EphemeralClient()` + `InMemoryStore` rebuilt every run | `PersistentClient("data/chroma")` + `data/parents.jsonl`, built once by `ingest` |
| `input()` loop + `IPython.display.Markdown` | Streamlit chat (`st.chat_input`, expanders for `<think>` / HyDE / raw output) |
| Kaggle/Colab secrets | `HF_TOKEN` env var (`.env`) |

The `<think>\n` prefill is kept: the prompt is rendered locally with the model's chat template and sent to
vLLM's `/v1/completions` (chat completions cannot prefill the assistant turn). HyDE's
`num_return_sequences=2` becomes `n=2`. Sampling params (`top_p=0.9`, `repetition_penalty=1.1`) match the notebook.

## Deploy (GPU VM: AWS g4dn / GCP T4 / L4 / Vast.ai / …)

Requirements on the host: NVIDIA driver, Docker, and the NVIDIA Container Toolkit
(`docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` must work).
The AWS Deep Learning AMI and GCP Deep Learning VM images already include all three.

```bash
git clone <this repo> && cd chatbot-finetuned-Qwen2
cp .env.example .env          # set HF_TOKEN if the model repo is private; tune VLLM_GPU_MEMORY_UTILIZATION
docker compose up -d --build
docker compose logs -f        # first start: vLLM downloads ~6 GB, app downloads 4 PDFs and embeds 7,008 child chunks
```

Open `http://<host>:8501`. To keep the port private, tunnel instead of opening the firewall:
`ssh -L 8501:localhost:8501 <host>`.

The first start of `app` runs `python -m legal_rag.ingest --skip-if-exists`. Later restarts reuse `./data`.
To rebuild the index (e.g. after changing chunking): `docker compose exec app python -m legal_rag.ingest`,
then `docker compose restart app`.

### Verify

1. Ask the required test case — *"Saya staf admin, kemarin lembur 3 jam untuk beresin laporan. Apakah saya
   berhak dapat uang lembur?"* — and check the `<think>` block, the answer, and the citation table.
2. Ask *"Menurut PP 35/2021, …"* → caption shows filter `{'peraturan_id': 'PP-35-2021'}`.
3. Ask *"Siapa pemenang Piala Dunia 2022?"* → route `internet (DuckDuckGo)`.
4. Re-check the threshold on this hardware:
   ```bash
   docker compose exec app python -m legal_rag.calibrate
   ```
   If it reports a new value, put it in `.env` as `RERANK_THRESHOLD=…` and `docker compose up -d app`.

### GPU memory

| GPU | `VLLM_GPU_MEMORY_UTILIZATION` | vLLM share | left for bge-m3 + reranker |
|---|---|---|---|
| T4 16 GB | 0.6 | ~9.6 GB (6.2 GB weights + KV cache) | ~6 GB |
| L4 24 GB | 0.5 | ~12 GB | ~12 GB |

`vllm` starts first (`depends_on: service_healthy`) so it reserves its share before the app loads its models.
To run embeddings + reranker on CPU instead, delete the `deploy:` block under `app` (queries and the first ingest become much slower).

## Local development (macOS, no NVIDIA GPU)

vLLM needs an NVIDIA GPU, so on a Mac run only the app and point it at a remote vLLM:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ssh -L 8000:localhost:8000 <gpu-host>       # vLLM from the GPU VM
python -m legal_rag.ingest                   # embeds on MPS/CPU, writes ./data
streamlit run streamlit_app.py
```
