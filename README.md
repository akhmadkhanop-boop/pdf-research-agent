# 📄 PDF Research Agent (RAG System)

An autonomous AI research assistant built to read, analyze, and extract precise information from PDF documents using **Retrieval-Augmented Generation (RAG)**. 

Instead of feeding entire lengthy documents to an LLM, this agent intelligently chunks the text, stores it in a local vector database, and retrieves only the most relevant sections to answer user queries accurately.

## 🚀 Features
* **Document Parsing:** Extracts raw text from PDF files using `pypdf`.
* **Smart Chunking:** Breaks down large texts into digestible chunks (500 characters) for optimal embedding.
* **Vector Storage:** Utilizes `ChromaDB` for fast and efficient local vector search.
* **Embeddings:** Powered by HuggingFace's `SentenceTransformers` (all-MiniLM-L6-v2) to understand the context of the text.
* **LLM Integration:** Connects with `Groq` (Llama-3.3-70b) via `LangChain`/`LangGraph` for lightning-fast and accurate answers based *strictly* on the provided PDF.

## 🛠️ Tech Stack
* **Language:** Python
* **LLM Framework:** LangChain / LangGraph
* **Vector Database:** ChromaDB
* **LLM Provider:** Groq API
* **Embeddings:** Sentence-Transformers

## ⚙️ Setup & Installation

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/pdf-research-agent.git](https://github.com/your-username/pdf-research-agent.git)
cd pdf-research-agent
