import os
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.rag.embeddings import create_embeddings

logger = logging.getLogger(__name__)


def get_vector_store(pdf_path: str, faiss_path: str, embedding_provider: str, embedding_api_key: str):
    """根据传入配置加载或创建向量数据库。"""
    embeddings = create_embeddings(embedding_provider, embedding_api_key)
    os.makedirs(os.path.dirname(faiss_path), exist_ok=True)

    if os.path.exists(faiss_path):
        logger.info("正在从本地加载向量数据库: %s", faiss_path)
        return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

    logger.info("本地未找到向量数据库，正在基于上传的PDF创建新的数据库...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_split_docs = text_splitter.split_documents(docs)

    vector_store = FAISS.from_documents(all_split_docs, embeddings)
    logger.info("向量数据库创建成功，正在保存到本地...")
    vector_store.save_local(faiss_path)
    logger.info("向量数据库已保存至: %s", faiss_path)
    return vector_store
