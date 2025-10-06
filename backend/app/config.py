import os

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "llama-3.1-70b-versatile")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    STORAGE_DIR: str = os.getenv("STORAGE_DIR", "./storage/index")
    DOCS_BASE_URL: str = os.getenv("DOCS_BASE_URL", "")
    ALLOWED_ORIGINS: str = os.getenv("ALLOWED_ORIGINS", "*")
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    REGION_FILTERS: str = os.getenv("REGION_FILTERS", "")

settings = Settings()