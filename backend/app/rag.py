from typing import Dict, Any, List
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from .config import settings
from .store import load_vectorstore, save_vectorstore

SYSTEM_PROMPT = """You are a precise HR policy assistant. Answer only from provided policy context.
- Cite at least one source section with doc id and section header if present.
- If the answer is not clearly in the context, say "I don't have that in policy" and suggest contacting HR.
- Keep answers under 200 words unless asked for details.
- Use exact dates/numbers when present. Prefer effective dates and caveats.
- Never invent numbers or dates. Prefer exact language and effective dates.
Return JSON strictly matching:
{"answer": "...", "citations": [{"doc_id": "...", "section": "...", "snippet": "...", "page": 0}], "policy_matches": ["..."], "confidence": "low|medium|high", "follow_up_suggestions": ["..."], "disclaimer": "..."}
"""

ANSWER_PROMPT = PromptTemplate.from_template(
"""System:
{system}

Context:
{context}

User question: {question}

JSON:"""
)

def get_llm():
    return ChatGroq(
        groq_api_key=settings.GROQ_API_KEY,
        model_name=settings.MODEL_NAME,
        temperature=0
    )

def ensure_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)

def ensure_vectorstore() -> FAISS:
    vs = load_vectorstore()
    if vs is None:
        raise RuntimeError("Vector store not found. Run ingestion first.")
    return vs

def retrieve(query: str, top_k: int, filters: Dict[str, Any] | None) -> List[Document]:
    vs = ensure_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": top_k})
    if filters and "region" in filters:
        region = filters["region"]
        retriever.search_kwargs["filter"] = lambda m: m.get("region") in [region, None]
    return retriever.get_relevant_documents(query)

def compose_context(docs: List[Document]) -> str:
    blocks = []
    for d in docs:
        meta = d.metadata or {}
        section = meta.get("section") or meta.get("heading")
        doc_id = meta.get("doc_id") or meta.get("source") or "unknown"
        page = meta.get("page")
        blocks.append(f"[{doc_id} | section: {section} | page: {page}]\n{d.page_content}")
    return "\n\n".join(blocks)

def generate_answer(question: str, docs: List[Document]) -> Dict[str, Any]:
    llm = get_llm()
    chain = LLMChain(llm=llm, prompt=ANSWER_PROMPT)
    ctx = compose_context(docs)
    raw = chain.run(system=SYSTEM_PROMPT, context=ctx, question=question)
    import json
    try:
        data = json.loads(raw)
    except Exception:
        data = {"answer": raw, "citations": [], "policy_matches": [], "confidence": "medium", "follow_up_suggestions": [], "disclaimer": "Please verify with HR for official confirmation."}
    return data