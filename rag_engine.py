from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict
from scraper import DataScraper

class RAGEngine:
    """
    PDF RAG System — LangChain + Groq + FAISS
    FastEmbed embeddings — no torch/transformers conflict
    """

    def __init__(self, api_key: str, chunk_size: int = 500):
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.docs = []
        self.vectorstore = None
        self.chain = None
        self.retriever = None
        self.scraper = DataScraper()

        # Groq LLM
        self.llm = ChatGroq(
            api_key=api_key,
            model_name="llama-3.1-8b-instant",
            temperature=0.2,
        )

        # FastEmbed — lightweight, no torch conflict
        self.embeddings = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5"
        )

        # Text splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " "],
        )

    def load_pdfs(self, pdf_paths: List[str]) -> None:
        all_docs = []
        for path in pdf_paths:
            loader = PyPDFLoader(path)
            pages = loader.load()
            all_docs.extend(pages)
        self.docs = self.splitter.split_documents(all_docs)
        print(f"Loaded {len(self.docs)} chunks from {len(pdf_paths)} PDF(s)")

    def load_wikipedia(self, query: str) -> None:
        wiki_docs = self.scraper.scrape_wikipedia(query)
        if wiki_docs:
            split_wiki = self.splitter.split_documents(wiki_docs)
            self.docs.extend(split_wiki)
            print(f"Loaded {len(split_wiki)} chunks from Wikipedia for '{query}'")

    def load_url(self, url: str) -> None:
        url_docs = self.scraper.scrape_url(url)
        if url_docs:
            split_url = self.splitter.split_documents(url_docs)
            self.docs.extend(split_url)
            print(f"Loaded {len(split_url)} chunks from URL '{url}'")

    def build_vectorstore(self) -> None:
        if not self.docs:
            raise ValueError("Please load a document first (PDF, Wiki, or URL)!")
            
        print(f"Total {len(self.docs)} chunks ready for embedding. Building vector store...")
        
        # Add data in small batches to avoid OOM/Memory errors
        batch_size = 50
        for i in range(0, len(self.docs), batch_size):
            batch = self.docs[i:i + batch_size]
            if i == 0:
                self.vectorstore = FAISS.from_documents(batch, self.embeddings)
            else:
                self.vectorstore.add_documents(batch)
            print(f"Batch {i//batch_size + 1} processed ({len(batch)} chunks)")
            
        print("Vector store ready!")

    def build_chain(self, top_k: int = 3) -> None:
        if not self.vectorstore:
            raise ValueError("Call build_vectorstore() first!")

        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

        prompt = ChatPromptTemplate.from_template("""You are a helpful AI assistant that answers questions from PDF documents and scraped web/Wikipedia sources.

The context may include both PDF and web sources. Cross-reference data from different sources to provide a complete and accurate answer.
If different sources agree with each other or explain each other, mention it.
If the answer is not in the context, clearly state "This information is not in the document."
Keep the answer clear and concise.

Context:
{context}

Question: {question}

Answer:""")

        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        self.chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        print("RAG chain ready!")

    def query(self, question: str) -> Dict:
        if not self.chain:
            raise ValueError("Call build_chain() first!")

        answer = self.chain.invoke(question)

        source_docs = self.retriever.invoke(question)
        sources = []
        for doc in source_docs:
            source_info = doc.metadata.get("source", "Unknown Source")
            page = doc.metadata.get("page")
            
            if page is not None:
                try:
                    source_label = f"Source: {source_info} (Page {int(page) + 1})"
                except ValueError:
                    source_label = f"Source: {source_info} (Page {page})"
            else:
                source_label = f"Source: {source_info}"
                
            snippet = doc.page_content[:200].strip().replace("\n", " ")
            sources.append(f"{source_label} | {snippet}...")

        return {
            "answer": answer,
            "sources": sources,
        }
