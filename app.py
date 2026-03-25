import streamlit as st
from rag_engine import RAGEngine
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="PDF RAG System",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
.main-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
.sub-title { color: #666; font-size: 1rem; margin-bottom: 1.5rem; }
.chat-user { background: #e8f4fd; padding: 12px 16px; border-radius: 12px; margin: 8px 0; }
.chat-bot { background: #f0f9f0; padding: 12px 16px; border-radius: 12px; margin: 8px 0; border-left: 3px solid #2ecc71; }
.source-box { background: #fff8e1; padding: 8px 12px; border-radius: 8px; font-size: 0.85rem; color: #555; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Load from .env first, allow manual override
    env_key = os.getenv("GROQ_API_KEY", "")
    groq_key = st.text_input(
        "Groq API Key",
        value=env_key,
        type="password",
        placeholder="gsk_...",
        help="Create a free account on groq.com"
    )
    if env_key:
        st.success("✅ API Key loaded from .env")
    
    st.markdown("---")
    st.markdown("### 📄 PDF Upload")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    st.markdown("---")
    st.markdown("### 🌐 Web & Wiki Data")
    wiki_topic = st.text_input("Wikipedia Topic (optional)", placeholder="e.g. Artificial Intelligence")
    web_url = st.text_input("Website URL (optional)", placeholder="e.g. https://example.com")
    
    st.markdown("---")
    chunk_size = st.slider("Chunk size", 200, 1000, 500, 100,
                           help="Larger = more context, smaller = more precise")
    top_k = st.slider("Top K results", 1, 8, 3,
                      help="Number of chunks to build the answer")
    
    process_btn = st.button("🚀 Process PDFs", use_container_width=True)
    
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Enter Groq API key")
    st.markdown("2. Upload PDF, enter Wiki topic or URL")
    st.markdown("3. Click 'Process'")
    st.markdown("4. Ask questions!")

# ── Session State ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag" not in st.session_state:
    st.session_state.rag = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# ── Main Area ─────────────────────────────────────────────
st.markdown('<div class="main-title">📚 PDF RAG System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Intelligent Q&A from your PDFs — powered by LangChain + Groq</div>', unsafe_allow_html=True)

# Process PDFs
if process_btn:
    if not groq_key:
        st.error("❌ Enter Groq API key first!")
    elif not uploaded_files and not wiki_topic and not web_url:
        st.error("❌ Please provide a PDF, Wiki topic, or URL!")
    else:
        with st.spinner("Processing PDFs... ⏳"):
            try:
                # Save uploaded PDFs temporarily
                tmp_paths = []
                os.makedirs("/tmp/rag_pdfs", exist_ok=True)
                for f in uploaded_files:
                    path = f"/tmp/rag_pdfs/{f.name}"
                    with open(path, "wb") as fp:
                        fp.write(f.read())
                    tmp_paths.append(path)
                
                # Initialize RAG engine
                rag = RAGEngine(api_key=groq_key, chunk_size=chunk_size)
                
                if uploaded_files:
                    rag.load_pdfs(tmp_paths)
                if wiki_topic:
                    rag.load_wikipedia(wiki_topic)
                if web_url:
                    rag.load_url(web_url)
                
                if not rag.docs:
                     raise ValueError("Could not extract any valid data!")

                rag.build_vectorstore()
                rag.build_chain(top_k=top_k)
                
                st.session_state.rag = rag
                st.session_state.pdf_processed = True
                st.session_state.messages = []
                
                st.success("✅ Data source(s) processed! You can ask questions now.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# ── Chat Interface ─────────────────────────────────────────
if st.session_state.pdf_processed and st.session_state.rag:
    st.markdown("---")
    
    # Display chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("sources"):
                with st.expander("📄 View Sources"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f'<div class="source-box">**Source {i}:** {src}</div>', unsafe_allow_html=True)
    
    # Input
    user_input = st.chat_input("Ask something about the document or data...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Thinking... 🤔"):
            try:
                result = st.session_state.rag.query(user_input)
                answer = result["answer"]
                sources = result["sources"]
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"❌ Error occurred: {str(e)}"
                })
        
        st.rerun()

elif not st.session_state.pdf_processed:
    st.info("👈 Upload PDF from the left sidebar and click process!")
    
    # Demo cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📖 What can it do?")
        st.markdown("- Answers from PDF, Wiki, and Web\n- Cross-reference multiple sources\n- Answers with sources")
    with col2:
        st.markdown("### ⚡ Technology")
        st.markdown("- LangChain RAG\n- Groq LLM (free)\n- FAISS Vector Store")
    with col3:
        st.markdown("### 🎯 Best for")
        st.markdown("- Research papers\n- Books / notes\n- Company documents")
