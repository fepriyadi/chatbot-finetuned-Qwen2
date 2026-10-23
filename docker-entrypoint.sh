#!/bin/sh
set -e

# Start pertama: unduh PDF + embed ke ChromaDB (data/ di-mount sebagai volume, jadi hanya sekali).
python -m legal_rag.ingest --skip-if-exists

exec streamlit run streamlit_app.py \
    --server.address=0.0.0.0 \
    --server.port=8501 \
    --server.headless=true \
    --browser.gatherUsageStats=false
