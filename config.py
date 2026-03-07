import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
MEMORY_DB_PATH = os.path.join(os.path.dirname(__file__), "memory_store", "memory.db")
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "vector_store")
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "uploads")

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"
FAST_MODEL = "gpt-4o-mini"   # lightweight agents (guardrail, parser, router)
WHISPER_MODEL = "gpt-4o-transcribe"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5

os.makedirs(os.path.dirname(MEMORY_DB_PATH), exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
