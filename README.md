# 📚 PDF RAG System
**LangChain + Groq + FAISS + Streamlit**

Intelligent Q&A system from your PDF documents — completely free!

---

## ⚡ Quick Setup (5 minutes)

### Step 1 — Install
```bash
pip install -r requirements.txt
```

### Step 2 — Get Groq API Key (FREE)
1. Go to https://console.groq.com
2. Create a free account
3. API Keys → Create Key
4. Copy the key

### Step 3 — Run App
```bash
streamlit run app.py
```

### Step 4 — Usage
1. Enter your Groq API key in the sidebar
2. Upload PDF(s)
3. Click the "Process PDFs" button
4. Ask questions! 🎉

---

## 📁 Project Structure
```
rag_system/
├── app.py          ← Streamlit UI
├── rag_engine.py   ← RAG core logic
├── requirements.txt
└── README.md
```

---

## 🔧 RAG Flow
```
PDF Upload
    ↓
PyPDFLoader (text extract)
    ↓
RecursiveCharacterTextSplitter (chunks)
    ↓
FastEmbed Embeddings (free, local)
    ↓
FAISS Vector Store
    ↓
User Question → Similarity Search → Top K Chunks
    ↓
Groq LLaMA3 → Answer + Sources
```

---

## 💡 Tips
- **Chunk size 500** is best for most documents
- **Top K = 3** provides accurate answers
- You can upload multiple PDFs at once
