import os
from typing import Optional, List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from .config import settings

class SentenceTransformerEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

def get_embeddings():
    return SentenceTransformerEmbeddings(model_name=settings.EMBEDDING_MODEL)

def load_vectorstore() -> Optional[FAISS]:
    if not os.path.isdir(settings.STORAGE_DIR):
        return None
    try:
        return FAISS.load_local(settings.STORAGE_DIR, get_embeddings(), allow_dangerous_deserialization=True)
    except Exception:
        return None

def save_vectorstore(vs: FAISS):
    os.makedirs(settings.STORAGE_DIR, exist_ok=True)
    vs.save_local(settings.STORAGE_DIR)

def build_docs(raws: List[Dict[str, Any]]) -> List[Document]:
    docs = []
    for r in raws:
        meta = r.get("metadata", {})
        page = meta.get("page")
        docs.append(Document(page_content=r["text"], metadata=meta))
    return docs