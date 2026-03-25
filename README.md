# 📚 PDF RAG System
**LangChain + Groq + FAISS + Streamlit**

Apne PDF documents se intelligent Q&A system — bilkul free!

---

## ⚡ Quick Setup (5 minutes)

### Step 1 — Install karo
```bash
pip install -r requirements.txt
```

### Step 2 — Groq API Key lo (FREE)
1. https://console.groq.com par jao
2. Free account banao
3. API Keys → Create Key
4. Copy karo

### Step 3 — App chalao
```bash
streamlit run app.py
```

### Step 4 — Use karo
1. Sidebar mein Groq API key daalo
2. PDF(s) upload karo
3. "PDFs Process Karo" button dabao
4. Sawaal poocho! 🎉

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
HuggingFace Embeddings (free, local)
    ↓
FAISS Vector Store
    ↓
User Question → Similarity Search → Top K Chunks
    ↓
Groq LLaMA3 → Answer + Sources
```

---

## 💡 Tips
- **Chunk size 500** zyada tar documents ke liye best hai
- **Top K = 3** accurate answers deta hai
- Multiple PDFs ek saath upload kar sakte ho
- Urdu/Roman Urdu mein bhi poocho, jawab milega!
