import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings,OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
#from agent import config

def get_vector_store(pdf_path, faiss_path, embedding_provider, embedding_api_key):
    """根据传入的配置加载或创建向量数据库"""
    if embedding_provider == "DashScope (Alibaba)":
        embeddings = DashScopeEmbeddings(
            model="text-embedding-v3", # DashScope的模型名
            dashscope_api_key=embedding_api_key,
        )
    elif embedding_provider == "OpenAI":
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", # OpenAI的模型名
            api_key=embedding_api_key
        )
    else:
        raise ValueError(f"不支持的嵌入模型提供商: {embedding_provider}")

    if os.path.exists(faiss_path):
        print(f"正在从本地加载向量数据库: {faiss_path}")
        return FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print("本地未找到向量数据库，正在基于上传的PDF创建新的数据库...")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_split_docs = text_splitter.split_documents(docs)
        
        vector_store = FAISS.from_documents(all_split_docs, embeddings)
        print("向量数据库创建成功，正在保存到本地...")
        vector_store.save_local(faiss_path)
        print(f"向量数据库已保存至: {faiss_path}")
        return vector_store


def create_rag_tool(vector_store):
    """创建一个使用特定向量数据库的RAG工具"""
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
    
    return [read_instructions]