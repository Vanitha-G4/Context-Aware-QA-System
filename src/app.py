
"""
app.py
------
Streamlit UI layer. No RAG logic lives here — everything document/LLM
related is delegated to backend.py so the two concerns stay separate.
"""
 
import streamlit as st
import backend
 
st.set_page_config(page_title="RetrieVia", layout="wide")
 
# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    body {background-color: #0e1117;}
    .block-container {padding-top: 2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)
 
# ---------------------------------------------------------------------------
# Sidebar: upload + reset
# ---------------------------------------------------------------------------
st.sidebar.title("📄 RetrieVia")
st.sidebar.markdown("AI Document Assistant")
 
uploaded_file = st.sidebar.file_uploader("Upload PDF Document", type="pdf")
 
if st.sidebar.button("🔄 Reset"):
    st.session_state.clear()
    st.rerun()
 
# ---------------------------------------------------------------------------
# Indexing (runs once per uploaded file)
# ---------------------------------------------------------------------------
if uploaded_file:
    if "vectorstore" not in st.session_state:
        with st.spinner("🔄 Processing document..."):
            st.session_state.vectorstore = backend.create_index(uploaded_file)
        st.sidebar.success("✅ Document Ready!")
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
st.title("🤖 Document Intelligence Assistant")
st.markdown("Ask questions from your uploaded document")
 
if "vectorstore" not in st.session_state:
    st.warning("📂 Please upload a PDF to start")
else:
    user_input = st.chat_input("Ask your question...")
 
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
 
        with st.spinner("Thinking..."):
            answer, docs = backend.rag_response(
                st.session_state.vectorstore,
                user_input,
                st.session_state.messages,
            )
 
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": docs}
        )
 
# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.write(msg["content"])
            with st.expander("Sources"):
                for i, doc in enumerate(msg.get("sources", [])):
                    st.markdown(f"**Source {i + 1}:**")
                    st.write(doc.page_content)