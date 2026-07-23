"""
backend.py
----------
Core RAG logic, split into two clear phases:
 
1. INDEXING  (create_index)   -> load PDF, split into chunks, embed, store in FAISS
2. QUERY     (rag_response)   -> retrieve relevant chunks, build prompt, call LLM
"""
 
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
 
 
# ---------------------------------------------------------------------------
# Embeddings (free, local, via sentence-transformers)
# ---------------------------------------------------------------------------
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
 
 
# ---------------------------------------------------------------------------
# INDEXING PHASE
# Upload -> Load -> Split -> Embed -> Store
# ---------------------------------------------------------------------------
def create_index(uploaded_file, temp_path: str = "temp.pdf"):
    """Build a FAISS vector store from an uploaded PDF file."""
 
    # 1. Save upload to disk so PyPDFLoader can read it
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
 
    try:
        # 2. Load
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
 
        # 3. Split into overlapping chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        docs = splitter.split_documents(documents)
 
        # 4. Embed + store
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)
 
        return vectorstore
    finally:
        # Clean up temp file regardless of success/failure
        if os.path.exists(temp_path):
            os.remove(temp_path)
 
 
# ---------------------------------------------------------------------------
# LLM (free, local, via Ollama)
# ---------------------------------------------------------------------------
def get_llm():
    return Ollama(model="llama3")
 
 
# ---------------------------------------------------------------------------
# QUERY PHASE
# Question -> Retrieve -> Build prompt (context + history) -> Generate
# ---------------------------------------------------------------------------
def format_history(chat_history, max_turns: int = 6):
    """Turn the last N chat turns into a plain-text transcript for the prompt."""
    recent = chat_history[-max_turns:]
    lines = [f"{msg['role']}: {msg['content']}" for msg in recent]
    return "\n".join(lines)
 
 
def rag_response(vectorstore, question, chat_history):
    # 1. Retrieve top-k relevant chunks
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
 
    # 2. Format recent chat history (excludes the current question, since
    #    it's appended to session_state.messages before this is called)
    history_text = format_history(chat_history[:-1])
 
    # 3. Build the prompt (history is now actually included)
    prompt = f"""You are a helpful assistant answering questions about an uploaded document.
 
Conversation so far:
{history_text if history_text else "(no previous messages)"}
 
Document context:
{context}
 
Question:
{question}
 
Answer clearly and concisely using only the document context above. If the
context doesn't contain the answer, say so instead of guessing.
"""
 
    # 4. Generate
    llm = get_llm()
    response = llm.invoke(prompt)
 
    return response, docs