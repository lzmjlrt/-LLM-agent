import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from agent import config

def get_vector_store():
    """加载或创建向量数据库"""
    embeddings = DashScopeEmbeddings(
        model=config.EMBEDDING_MODEL_NAME,
        dashscope_api_key=config.DASHSCOPE_API_KEY,
    )

    if os.path.exists(config.FAISS_INDEX_PATH):
        print(f"正在从本地加载向量数据库: {config.FAISS_INDEX_PATH}")
        return FAISS.load_local(config.FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print("本地未找到向量数据库，正在创建新的数据库...")
        loader = PyPDFLoader(config.PDF_PATH)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_split_docs = text_splitter.split_documents(docs)
        
        vector_store = FAISS.from_documents(all_split_docs, embeddings)
        print("向量数据库创建成功，正在保存到本地...")
        vector_store.save_local(config.FAISS_INDEX_PATH)
        print(f"向量数据库已保存至: {config.FAISS_INDEX_PATH}")
        return vector_store

# 初始化并获取向量数据库
vector_store = get_vector_store()

@tool
def read_instructions(user_review: str) -> str:
    """
    当用户评论中提到具体产品不会使用的问题时，调用此工具从说明书PDF中检索信息并返回相关内容。
    """
    print(f"--- [工具执行] 正在为问题 '{user_review}' 检索说明书 ---")
    similar_docs = vector_store.similarity_search(user_review, k=2)
    doc_input = ""
    for i, doc in enumerate(similar_docs):
        doc_input += f"文档{i+1}：{doc.page_content}\n\n"
    return doc_input if doc_input else "在说明书中未找到相关内容。"

# 将所有工具放入一个列表，方便导入
tools = [read_instructions]