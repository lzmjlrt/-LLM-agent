from agent.errors import ConfigValidationError

SUPPORTED_LLM_PROVIDERS = ("DeepSeek", "OpenAI")
SUPPORTED_EMBEDDING_PROVIDERS = ("DashScope (Alibaba)", "OpenAI")


def validate_runtime_config(
    llm_provider: str,
    llm_api_key: str,
    uploaded_file,
    embedding_provider: str,
    embedding_api_key: str,
):
    """在初始化前校验运行配置。"""
    if llm_provider not in SUPPORTED_LLM_PROVIDERS:
        raise ConfigValidationError(f"不支持的 LLM 提供商: {llm_provider}", "请选择 DeepSeek 或 OpenAI。")
    if embedding_provider not in SUPPORTED_EMBEDDING_PROVIDERS:
        raise ConfigValidationError(f"不支持的嵌入模型提供商: {embedding_provider}", "请选择 DashScope (Alibaba) 或 OpenAI。")

    if not llm_api_key or not llm_api_key.strip():
        raise ConfigValidationError("LLM API Key 不能为空。", f"请填写 {llm_provider} API Key。")
    if not embedding_api_key or not embedding_api_key.strip():
        raise ConfigValidationError("Embedding API Key 不能为空。", f"请填写 {embedding_provider} API Key。")

    if uploaded_file is None:
        raise ConfigValidationError("请先上传产品说明书 PDF。", "上传后再点击“应用配置”。")
    file_name = (getattr(uploaded_file, "name", "") or "").strip()
    if not file_name.lower().endswith(".pdf"):
        raise ConfigValidationError("上传文件格式不正确。", "仅支持 .pdf 文件。")

    file_size = len(uploaded_file.getbuffer())
    if file_size == 0:
        raise ConfigValidationError("上传的 PDF 为空文件。", "请上传有效的说明书 PDF。")
