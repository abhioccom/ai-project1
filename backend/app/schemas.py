from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class AskRequest(BaseModel):
    question: str
    filters: Optional[Dict[str, Any]] = {}
    top_k: Optional[int] = None
    follow_up_context: Optional[str] = None

class Citation(BaseModel):
    doc_id: str
    section: str
    snippet: str
    page: Optional[int] = None
    url: Optional[str] = None

class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    policy_matches: List[str]
    confidence: str
    follow_up_suggestions: List[str]
    disclaimer: Optional[str] = None
    metadata: Dict[str, Any]

class FeedbackRequest(BaseModel):
    answer_id: str
    helpful: bool
    comment: Optional[str] = None

class IngestResponse(BaseModel):
    message: str
    documents_processed: int
    chunks_created: int