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
    build_sft_data.py  RAG worked-example dataset for a short SFT on top of the GRPO model
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
`num_return_sequences=2` becomes `n=2`. Sampling params (`top_p=0.9`, `repetition_penalty=1.0`) match the notebook.

`RAG_PROMPT` makes the model quote the relevant pasal (with its `[n]`) inside `<think>` before concluding, and
`repetition_penalty` is 1.0 because a penalty above 1 also counts prompt tokens, so copying the context was penalised.
Without both, the model paraphrased PP 35/2021 Pasal 26 ayat (1) "paling lama 4 jam" (at most) as "4 jam atau lebih"
(at least) and told the 3-hour overtime test case it was not entitled, although Pasal 27 ayat (1) requires payment.

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

## RAG SFT dataset

The prompt fix made the model quote PP 35/2021 Pasal 26 ayat (1) correctly, but it still concluded
"3 jam tidak termasuk dalam batas". Neither training stage ever showed it a RAG prompt (GRPO kept prompts
≤ 256 tokens), so `legal_rag.build_sft_data` builds worked examples from the four PDFs:

1. parent chunks with a numeric limit ("paling lama 5 (lima) tahun"); chunks mentioning "lembur" are held out
   because the overtime question is the test case
2. the teacher, open-source `Qwen/Qwen2.5-32B-Instruct-AWQ` served by vLLM with JSON-schema output, copies the
   right/obligation, limit, and exception sentences verbatim and writes questions inside the limit, past it, and
   hitting the exception
3. each question goes through the real retrieval (filter → HyDE → ensemble → rerank Top-3)
4. the `<think>` target is assembled by code: quotes with `[n]`, `fakta ≤ batas → …`, conclusion, `Jawaban: Ya/Tidak`
5. an example is dropped unless every quote is verbatim in the context, `[n]` and the pasal match, and the
   label agrees with the numbers

The 32B teacher (~19 GB in 4-bit) and the GRPO model don't fit on Kaggle's 2×T4 at the same time, so the run is
split into two stages:

```bash
python -m legal_rag.build_sft_data --dry-run                 # candidate chunks only

# stage 1: teacher (steps 1-2), on port 8001
vllm serve Qwen/Qwen2.5-32B-Instruct-AWQ --port 8001 --tensor-parallel-size 2 --quantization awq \
    --dtype half --max-model-len 8192 --gpu-memory-utilization 0.92
python -m legal_rag.build_sft_data --stage teacher --limit 10   # trial: check the yield first
python -m legal_rag.build_sft_data --stage teacher

# stage 2: stop the teacher, start the GRPO model on :8000 (HyDE), then steps 3-5
python -m legal_rag.build_sft_data --stage build              # --no-hyde if the GRPO vLLM isn't running
```

`TEACHER_BASE_URL` / `TEACHER_MODEL` override the teacher endpoint. T4 has no bf16 and no Marlin kernels, hence
`--dtype half --quantization awq`. Output goes to `data/rag_sft/` (`train.jsonl`, `eval.jsonl`, `rejects.jsonl`,
`teacher_cache.jsonl`). The train/eval split is by source chunk, so variants of one rule never land on both sides.
Teacher results are cached, so a rerun only repeats the calls that failed.

## Local development (macOS, no NVIDIA GPU)

vLLM needs an NVIDIA GPU, so on a Mac run only the app and point it at a remote vLLM:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ssh -L 8000:localhost:8000 <gpu-host>       # vLLM from the GPU VM
python -m legal_rag.ingest                   # embeds on MPS/CPU, writes ./data
streamlit run streamlit_app.py
```
