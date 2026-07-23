# Architecture

## Overview

RetrieVia is a Retrieval-Augmented Generation (RAG) system with two phases:
an offline **indexing** phase (runs once per uploaded PDF) and an online
**query** phase (runs once per user question).

## Components

| Component | Technology | Role |
|---|---|---|
| Document loader | `PyPDFLoader` | Extracts text from PDF |
| Text splitter | `RecursiveCharacterTextSplitter` | Breaks text into 500-char chunks, 50-char overlap |
| Embedding model | `all-MiniLM-L6-v2` (HuggingFace) | Converts chunks/queries into 384-dim vectors |
| Vector store | FAISS | In-memory similarity search |
| Retriever | FAISS `as_retriever(k=3)` | Returns top-3 relevant chunks per query |
| LLM | Llama 3 via Ollama | Generates the final answer, local and free |
| UI | Streamlit | Chat interface, file upload, source display |

## Indexing phase

```
PDF upload → PyPDFLoader → RecursiveCharacterTextSplitter → MiniLM embeddings → FAISS index
```

Chunk size (500 chars) and overlap (50 chars) are tuned for short-to-medium
technical documents. Larger source documents or documents with long-range
context (e.g. contracts) may benefit from a larger chunk size (800–1000).

## Query phase

```
Question → FAISS retriever (k=3) → prompt assembly (context + last 6 turns of history) → Llama 3 → answer + sources
```

The prompt template instructs the model to answer only from retrieved
context and to say so explicitly when the answer isn't present, to reduce
hallucination.

## Known limitations

- **In-memory index**: the FAISS store is not persisted to disk — re-uploading
  is required after a restart. For persistence, swap in `FAISS.save_local()` /
  `load_local()`, or move to Chroma/Pinecone/Qdrant.
- **Single document at a time**: no multi-document or multi-user isolation.
- **No re-ranking step**: retrieval is pure vector similarity; a cross-encoder
  re-ranker would likely improve precision on ambiguous queries.
- **Local LLM quality ceiling**: Llama 3 (8B, via Ollama) is weaker than
  hosted frontier models on complex multi-hop reasoning over context.
