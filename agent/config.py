import os
from dotenv import load_dotenv
load_dotenv()
# --- API and Keys ---
DEEPSEEK_API_KEY = "输入你的deepseek的api"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DASHSCOPE_API_KEY = "输入你的 DashScope API Key"

# --- Models ---
LLM_MODEL_NAME = "deepseek-chat"
EMBEDDING_MODEL_NAME = "text-embedding-v3"
#这里用的是阿里的 DashScope Embeddings
# --- File Paths ---
# 使用 os.path.join 确保路径在不同操作系统下都正确
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 获取项目根目录 Review_rebuttle
PDF_PATH = os.path.join(BASE_DIR, "ES.pdf")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "emb")