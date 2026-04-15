import streamlit as st
import uuid

from agent.errors import AgentInitializationError, AgentRuntimeError, ConfigValidationError
from agent.factories.config_validation import validate_runtime_config
from agent.logging_utils import setup_logging
from agent.services.chat_service import initialize_agent_runtime, invoke_agent


def _render_error(err):
    st.error(err.user_message)
    if err.detail:
        with st.expander("错误详情", expanded=False):
            st.code(err.detail)


def run_streamlit_app():
    setup_logging()
    st.set_page_config(page_title="智能客服系统", page_icon="🤖", layout="wide")
    st.title("🤖 智能客服系统")

    with st.sidebar:
        st.header("⚙️ 系统配置")
        st.subheader("1. 大语言模型 (LLM)")
        llm_provider = st.selectbox("选择模型提供商", ["DeepSeek", "OpenAI"])
        llm_api_key = st.text_input(f"输入 {llm_provider} API Key", type="password")

        st.subheader("2. 知识库 (PDF)")
        uploaded_file = st.file_uploader("上传产品说明书 (PDF)", type="pdf")

        st.subheader("3. 嵌入模型")
        embedding_provider = st.selectbox("选择嵌入模型提供商", ["DashScope (Alibaba)", "OpenAI"])
        embedding_api_key = st.text_input(f"输入 {embedding_provider} API Key", type="password")

        if st.button("应用配置", use_container_width=True):
            try:
                validate_runtime_config(
                    llm_provider=llm_provider,
                    llm_api_key=llm_api_key,
                    uploaded_file=uploaded_file,
                    embedding_provider=embedding_provider,
                    embedding_api_key=embedding_api_key,
                )
                with st.spinner("正在初始化系统，请稍候..."):
                    app = initialize_agent_runtime(
                        llm_provider=llm_provider,
                        llm_api_key=llm_api_key,
                        uploaded_file=uploaded_file,
                        embedding_provider=embedding_provider,
                        embedding_api_key=embedding_api_key,
                    )
                    st.session_state.app = app
                    st.session_state.configured = True
                    st.session_state.thread_id = str(uuid.uuid4())
                    st.success("系统初始化成功！可以开始聊天了。")
                    st.session_state.messages = [{"role": "assistant", "content": "系统已就绪，请问有什么可以帮助您的吗？"}]
            except ConfigValidationError as err:
                _render_error(err)
            except AgentInitializationError as err:
                _render_error(err)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "您好！请在左侧侧边栏完成配置后，与我开始对话。"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        if not st.session_state.get("configured", False):
            st.error("请先在左侧侧边栏完成配置并点击'应用配置'按钮！")
            return

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                app = st.session_state.app
                thread_id = st.session_state.get("thread_id")
                if not thread_id:
                    thread_id = str(uuid.uuid4())
                    st.session_state.thread_id = thread_id
                try:
                    response, _request_id = invoke_agent(app, prompt, thread_id)
                    st.write(response)
                except AgentRuntimeError as runtime_err:
                    _render_error(runtime_err)
                    response = "抱歉，处理您的请求时出现异常，请稍后重试。"
        st.session_state.messages.append({"role": "assistant", "content": response})
