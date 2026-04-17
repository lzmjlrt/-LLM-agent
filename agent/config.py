import os
from dotenv import load_dotenv

load_dotenv()

# --- API and Keys ---
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Models ---
DEEPSEEK_MODEL_NAME = "deepseek-chat"
OPENAI_CHAT_MODEL_NAME = "gpt-4o"
DASHSCOPE_EMBEDDING_MODEL_NAME = "text-embedding-v3"
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# --- File Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(BASE_DIR, "ES.pdf")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "emb")
TEMP_DIR_NAME = "temp"
DEFAULT_SESSION_FAISS_DIR = "faiss_index"
TEMP_DIR_PATH = os.path.join(BASE_DIR, TEMP_DIR_NAME)
UPLOADS_DIR_NAME = "uploads"
UPLOADS_DIR_PATH = os.path.join(TEMP_DIR_PATH, UPLOADS_DIR_NAME)
FAISS_CACHE_DIR_NAME = "faiss_cache"
FAISS_CACHE_DIR_PATH = os.path.join(TEMP_DIR_PATH, FAISS_CACHE_DIR_NAME)
CONVERSATION_STORE_DIR_NAME = "conversation_store"
CONVERSATION_STORE_DIR_PATH = os.path.join(TEMP_DIR_PATH, CONVERSATION_STORE_DIR_NAME)
CONVERSATION_HISTORY_LIMIT = int(os.getenv("CONVERSATION_HISTORY_LIMIT", "8"))
SEMANTIC_CACHE_ENABLED = os.getenv("SEMANTIC_CACHE_ENABLED", "true").lower() in ("1", "true", "yes", "on")
SEMANTIC_CACHE_SIMILARITY_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_SIMILARITY_THRESHOLD", "0.92"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
