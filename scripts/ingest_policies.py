import os, glob
from typing import List
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

load_dotenv()

# Settings directly here
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
STORAGE_DIR = os.getenv("STORAGE_DIR", "./storage/index")

def load_one(path: str) -> List[Document]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata["doc_id"] = os.path.basename(path)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(path)
        docs = loader.load()
        for d in docs:
            d.metadata["doc_id"] = os.path.basename(path)
    else:
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["doc_id"] = os.path.basename(path)
    for d in docs:
        if "section" not in d.metadata:
            first = d.page_content.split("\n")[0][:120]
            d.metadata["section"] = first.strip()
    return docs

class SentenceTransformerEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

def main():
    # Check if Groq API key is set
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not found in environment variables!")
        print("Please set your Groq API key in the .env file")
        return
    
    paths = glob.glob("./policies/*.*")
    if not paths:
        print("No policy files found in ./policies/")
        return
    
    print(f"Found {len(paths)} policy files")
    
    docs: List[Document] = []
    for p in paths:
        print(f"Loading {p}")
        docs.extend(load_one(p))
    
    print(f"Split into {len(docs)} documents")
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    
    print(f"Created {len(chunks)} chunks")
    
    print("Creating embeddings using SentenceTransformers...")
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    vs = FAISS.from_documents(chunks, embeddings)
    
    os.makedirs(STORAGE_DIR, exist_ok=True)
    vs.save_local(STORAGE_DIR)
    
    print(f"✅ Successfully indexed {len(chunks)} chunks from {len(paths)} files into {STORAGE_DIR}")

if __name__ == "__main__":
    main()