from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.agents import create_agent
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import chromadb
import logging
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PDF_Agent")

# ===== SETUP =====
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
client = chromadb.Client()
collection = client.create_collection("pdf_data")

# ===== READ PDF =====
def read_pdf(file_path):
    """Extracts all text from the PDF."""
    logger.info(f"Reading PDF: {file_path}")
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# ===== CHUNKING (large text → chunks) =====
def create_chunks(text, size=500):
    """Breaks large text into smaller chunks."""
    chunks = []
    for i in range(0, len(text), size):
        chunks.append(text[i:i + size])
    return chunks

# ===== PDF LOAD + STORE =====
pdf_text = read_pdf("test.pdf")           # name of your PDF file
chunks = create_chunks(pdf_text)
logger.info(f"{len(chunks)} chunks created")

embeddings = embed_model.encode(chunks).tolist()
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[str(i) for i in range(len(chunks))]
)

# ===== RAG TOOL (Search from PDF) =====
@tool
def search_pdf(question: str) -> str:
    """Finds relevant answers from the PDF content."""
    logger.info(f"PDF search: {question}")
    question_emb = embed_model.encode([question]).tolist()
    result = collection.query(query_embeddings=question_emb, n_results=3)
    return " ".join(result['documents'][0])

# ===== AI + AGENT =====
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=0.3)

system_prompt = """You are a PDF research assistant.
- Use the 'search_pdf' tool to answer from the PDF
- Give clear answers in English
- Answer ONLY using the PDF data"""

agent = create_agent(llm, [search_pdf], system_prompt=system_prompt)

# ===== RUN =====
def ask(question):
    try:
        result = agent.invoke({"messages": [("user", question)]})
        return result["messages"][-1].content
    except Exception as e:
        logger.error(f"Error: {e}")
        return "System is busy at the moment."

question = "What is written in this PDF? Give a summary."
print(f"\nQUESTION: {question}\n")
print("ANSWER:", ask(question))