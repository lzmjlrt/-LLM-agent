import sys
import os

import io
from langchain_community.chat_models import  ChatOpenAI
from langchain.chat_models import init_chat_model
#from agent.rag_setup import get_vector_store, create_rag_tool
import streamlit as st
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
from agent.graph_workflow import create_graph
from rag_setup import get_vector_store, create_rag_tool

                
def run_agent(review_text: str):
    """
    运行智能评论回复 Agent
    """
    # 创建并编译图
    app = create_graph()
    
    # 准备输入
    inputs = {"original_review": review_text}
    
    # 运行 Agent
    print(f"\n--- 开始处理新评论: '{review_text}' ---")
    result = app.invoke(inputs)
    
    # 打印最终结果
    print("\n--- Agent 最终回复 ---")
    print(result["finally_reply"])
    return result


if __name__ == "__main__":
    # # --- 在这里测试你的 Agent ---
    
    # # 测试用例1: 负面评价，需要调用工具
    # negative_review_with_tool = "这个产品太差了，我不知道怎么用！请帮我看看说明书。"
    
    # run_agent(negative_review_with_tool)
    
    # print("\n" + "="*50 + "\n")

    # # 测试用例2: 负面评价，无需调用工具
    # negative_review_no_tool = "物流太慢了，等了半个月才到！差评！"
    # run_agent(negative_review_no_tool)

    # print("\n" + "="*50 + "\n")

    # # 测试用例3: 正面评价
    # positive_review = "质量很好，非常喜欢！"
    # run_agent(positive_review)
    
    st.set_page_config(page_title="智能客服系统", page_icon="🤖", layout="wide")
    st.title("🤖 智能客服系统")

    with st.sidebar:
        st.header("⚙️ 系统配置")

        # 1. LLM 配置
        st.subheader("1. 大语言模型 (LLM)")
        llm_provider = st.selectbox("选择模型提供商", ["DeepSeek", "OpenAI"])
        llm_api_key = st.text_input(f"输入 {llm_provider} API Key", type="password")

        # 2. RAG (知识库) 配置
        st.subheader("2. 知识库 (PDF)")
        uploaded_file = st.file_uploader("上传产品说明书 (PDF)", type="pdf")
        
        st.subheader("3. 嵌入模型")
        embedding_provider = st.selectbox("选择嵌入模型提供商", ["DashScope (Alibaba)", "OpenAI"])
        embedding_api_key = st.text_input(f"输入 {embedding_provider} API Key", type="password")

        # 4. 应用配置按钮
        if st.button("应用配置", use_container_width=True):
            if not all([llm_api_key, uploaded_file, embedding_api_key]):
                st.error("请填写所有API Key并上传PDF文件！")
            else:
                with st.spinner("正在初始化系统，请稍候..."):
                    # 创建临时目录并保存上传的文件
                    temp_dir = "temp"
                    os.makedirs(temp_dir, exist_ok=True)
                    pdf_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(pdf_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # 1. 初始化 LLM
                    if llm_provider == "DeepSeek":
                        llm = init_chat_model(
                            "deepseek-chat",
                            model_provider="deepseek",
                            base_url="https://api.deepseek.com",
                            api_key= llm_api_key,
                            model_kwargs={"response_format": {"type": "json_object"}}
                        )
                    else: # OpenAI
                        llm = ChatOpenAI(model="gpt-4o", api_key=llm_api_key, temperature=0,model_kwargs={"response_format": {"type": "json_object"}})
                    
                    # 2. 初始化 RAG 工具
                    faiss_path = os.path.join(temp_dir, "faiss_index")
                    vector_store = get_vector_store(pdf_path, faiss_path, embedding_provider, embedding_api_key)
                    tools = create_rag_tool(vector_store)
                    
                    # 3. 创建并编译 Agent 图
                    app = create_graph(llm, tools)
                    
                    # 4. 将编译好的 app 存入 session_state
                    st.session_state.app = app
                    st.session_state.configured = True
                    st.success("系统初始化成功！可以开始聊天了。")
                    st.session_state.messages = [{"role": "assistant", "content": "系统已就绪，请问有什么可以帮助您的吗？"}]
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "您好！请在左侧侧边栏完成配置后，与我开始对话。"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if not st.session_state.get("configured", False):
            st.error("请先在左侧侧边栏完成配置并点击'应用配置'按钮！")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("思考中..."):
                    app = st.session_state.app
                    inputs = {"original_review": prompt}
                    result = app.invoke(inputs)
                    response = result.get("finally_reply", "抱歉，我无法处理您的请求。")
                    st.write(response)
            
            st.session_state.messages.append({"role": "assistant", "content": response})