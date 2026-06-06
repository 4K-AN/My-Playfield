# RAG Pipeline with Hybrid Search

A production-grade Retrieval-Augmented Generation system combining dense vector search (bi-encoder) and sparse BM25 retrieval, with cross-encoder reranking and a full evaluation suite.

## Architecture

```
Documents
    │
    ▼
┌─────────────┐
│   Chunker   │  ← Recursive char splitter + sliding window + sentence-aware
└──────┬──────┘
       │ chunks
       ▼
┌─────────────────────────────────────────┐
│              Index Layer                │
│  ┌──────────────┐   ┌────────────────┐  │
│  │  ChromaDB    │   │  BM25 Index    │  │
│  │ (dense vec)  │   │  (sparse tok)  │  │
│  └──────────────┘   └────────────────┘  │
└─────────────────────────────────────────┘
       │
  query at runtime
       │
       ▼
┌─────────────────────────────────────────┐
│           Hybrid Retriever              │
│  dense_results + sparse_results         │
│  fused via Reciprocal Rank Fusion (RRF) │
└────────────────────┬────────────────────┘
                     │
                     ▼
             ┌───────────────┐
             │ Cross-Encoder │  ← ms-marco reranker
             │   Reranker    │
             └───────┬───────┘
                     │
                     ▼
             ┌───────────────┐
             │  LLM Answer   │  ← OpenAI / local model
             └───────────────┘
                     │
                     ▼
             ┌───────────────┐
             │  Eval Suite   │  ← faithfulness + relevance
             └───────────────┘
```

## Stack

| Component | Library |
|-----------|---------|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Vector DB | `chromadb` |
| Sparse search | `rank_bm25` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `openai` (swap-able) |
| Eval | `ragas` + custom metrics |

## Setup

```bash
pip install -r requirements.txt
cp configs/config.example.yaml configs/config.yaml
# edit config.yaml with your API keys
python -m src.pipeline.ingest --docs data/
python -m src.pipeline.query --question "Your question here"
```

## Project Structure

```
rag-hybrid-search/
├── src/
│   ├── chunking/       # Document chunking strategies
│   ├── retrieval/      # Dense, sparse, hybrid retrieval
│   ├── reranking/      # Cross-encoder reranker
│   ├── pipeline/       # End-to-end ingest + query pipeline
│   └── eval/           # Faithfulness & relevance evaluation
├── tests/              # Unit + integration tests
├── configs/            # YAML config files
├── data/               # Sample documents
└── requirements.txt
```
