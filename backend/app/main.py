from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import time, os, httpx
from .schemas import AskRequest, AskResponse, IngestResponse, FeedbackRequest, Citation
from .config import settings
from .rag import retrieve, generate_answer
from .store import load_vectorstore
import glob
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer

app = FastAPI(title="AI Policy Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.ALLOWED_ORIGINS.split(",")] if settings.ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
async def healthz():
    ok = load_vectorstore() is not None
    return {"ok": ok}

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    t0 = time.time()
    docs = retrieve(req.question, req.top_k or settings.TOP_K, req.filters or {})
    data = generate_answer(req.question, docs)
    
    # enrich citations with URLs if available
    cits = []
    for c in data.get("citations", []):
        url = c.get("url")
        if not url and settings.DOCS_BASE_URL and c.get("doc_id"):
            url = f"{settings.DOCS_BASE_URL.rstrip('/')}/{c['doc_id']}"
        cits.append(Citation(**{**c, "url": url}))
    
    latency = int((time.time() - t0) * 1000)
    return AskResponse(
        answer=data.get("answer", ""),
        citations=cits,
        policy_matches=data.get("policy_matches", []),
        confidence=data.get("confidence", "medium"),
        follow_up_suggestions=data.get("follow_up_suggestions", []),
        disclaimer=data.get("disclaimer", "Please verify with HR for official confirmation."),
        metadata={"latency_ms": latency, "retriever_k": len(docs), "model": settings.MODEL_NAME},
    )

@app.post("/ingest", response_model=IngestResponse)
async def ingest(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Process files
    all_docs = []
    for file in files:
        if file.filename.endswith('.pdf'):
            loader = PyPDFLoader(file.filename)
            docs = loader.load()
        elif file.filename.endswith(('.docx', '.doc')):
            loader = Docx2txtLoader(file.filename)
            docs = loader.load()
        else:
            loader = TextLoader(file.filename)
            docs = loader.load()
        
        for d in docs:
            d.metadata["doc_id"] = file.filename
        all_docs.extend(docs)
    
    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(all_docs)
    
    # Create embeddings and store
    embeddings = SentenceTransformer(model_name=settings.EMBEDDING_MODEL)
    vs = FAISS.from_documents(chunks, embeddings)
    
    os.makedirs(settings.STORAGE_DIR, exist_ok=True)
    vs.save_local(settings.STORAGE_DIR)
    
    return IngestResponse(
        message=f"Successfully processed {len(files)} files",
        documents_processed=len(files),
        chunks_created=len(chunks)
    )

@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    os.makedirs("./storage", exist_ok=True)
    with open("./storage/feedback.tsv", "a", encoding="utf-8") as f:
        f.write(f"{req.answer_id}\t{req.helpful}\t{(req.comment or '').replace(chr(9),' ')}\n")
    return {"ok": True}

@app.get("/docs/{doc_id}")
async def get_doc(doc_id: str):
    return {"doc_id": doc_id, "title": doc_id, "url": f"{settings.DOCS_BASE_URL.rstrip('/')}/{doc_id}" if settings.DOCS_BASE_URL else None}

# WhatsApp webhook (Meta Cloud API)
@app.post("/webhooks/whatsapp")
async def whatsapp_webhook(request: Request):
    body = await request.json()
    try:
        entry = body["entry"][0]["changes"][0]["value"]
        phone_number_id = entry["metadata"]["phone_number_id"]
        msg = entry["messages"][0]
        from_wa = msg["from"]
        text = msg.get("text", {}).get("body", "")
    except Exception:
        return JSONResponse({"status": "ignored"})
    
    # Ask RAG
    result = await ask(AskRequest(question=text, filters={}, top_k=4))
    answer = result.answer
    
    # Reply via Meta send API
    token = os.getenv("WHATSAPP_TOKEN", "")
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(
            f"https://graph.facebook.com/v19.0/{phone_number_id}/messages",
            headers={"Authorization": f"Bearer {token}"},
            json={"messaging_product": "whatsapp", "to": from_wa, "type": "text", "text": {"preview_url": False, "body": answer[:4000]}},
        )
    return {"ok": True}