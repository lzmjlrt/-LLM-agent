# 智能评论回复 Agent

这是一个基于 LangChain 和 LangGraph 构建的智能客服项目，旨在自动分析电商平台上的用户评论，并根据评论内容生成高度定制化的回复。

## ✨ 项目特色

- **智能分诊**: 自动分析评论的情感（正面/负面/中性）和有效性（真实评论/默认好评）。
- **动态路由**: 使用 LangGraph 的条件路由，将不同类型的评论分发给不同的处理逻辑。
- **RAG 增强**: 对于提及“产品不会用”的负面评论，使用查询重写 + 向量检索 + BM25 混合检索 + rerank，从本地 `ES.pdf` 提取更相关信息。
- **工具调用**: 在负面评论处理流程中，Agent 被授权可以自主决定是否调用 `read_instructions` 工具来查询知识库。
- **会话记忆与语义缓存**: 基于 `thread_id` 持久化最近对话，并对高相似历史问题直接命中缓存回复。
- **模块化设计**: 代码结构按应用层、工作流层、RAG 层和工厂层拆分，便于维护和扩展。

## 🛠️ 技术栈

- **核心框架**: LangChain, LangGraph
- **大语言模型 (LLM)**: DeepSeek API
- **嵌入模型 (Embedding)**: 阿里巴巴通义千问 (DashScope)
- **向量数据库**: FAISS (本地)
- **数据模型**: Pydantic
- **环境管理**: python-dotenv

## 示意图
<img width="1584" height="854" alt="ca6afb01af6329c6a7f322672135b3b0" src="https://github.com/user-attachments/assets/2d864ea6-29e6-4601-841b-e3d544e5d2f1" />



## 📂 项目结构

```
.
├── agent/
│   ├── __init__.py
│   ├── main.py                # 运行入口（启动 Streamlit）
│   ├── __main__.py            # 包入口（python -m agent）
│   ├── config.py              # 配置中心（模型名、路径、环境变量）
│   ├── event_names.py         # 结构化事件名常量
│   ├── graph_workflow.py      # 兼容入口，导出 create_graph
│   ├── app/chat_app.py        # Streamlit UI 与会话编排
│   ├── factories/
│   │   ├── config_validation.py
│   │   └── model_factory.py   # LLM 初始化工厂
│   ├── services/
│   │   ├── chat_service.py        # 兼容 facade（对外导出）
│   │   ├── runtime_initializer.py # 运行时初始化（模型、索引、图）
│   │   ├── invoke_service.py      # 统一调用入口（含缓存/记忆）
│   │   ├── upload_store.py        # 上传文件持久化
│   │   └── conversation_memory.py # 会话记忆与语义缓存
│   ├── workflow/
│   │   ├── constants.py       # 意图/路由/状态常量
│   │   ├── intent_rules.py    # 意图与规则判定
│   │   ├── reply_policies.py  # 回复模板与回复校验
│   │   ├── node_factory.py    # LangGraph 节点组装
│   │   ├── nodes.py           # 兼容导出层
│   │   ├── router.py
│   │   ├── schema.py
│   │   └── graph.py
│   └── rag/
│       ├── cache.py
│       ├── embeddings.py
│       ├── vector_store.py
│       └── tools.py
├── tests/
│   ├── test_architecture_baseline.py
│   ├── test_workflow_router.py
│   ├── test_graph_smoke.py
│   ├── test_intent_rules.py
│   ├── test_reply_validation.py
│   ├── test_invoke_service.py
│   └── ...
├── .env                    # (需要手动创建) 存放API密钥
├── .gitignore              # Git忽略文件
├── ES.pdf                  # 产品说明书知识库
├── requirements.txt        # 项目依赖
└── README.md               # 项目说明
```

## 🚀 安装与配置

1.  **克隆项目**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **安装依赖**
    建议您先创建一个虚拟环境。然后运行以下命令安装所有必需的库：
    ```bash
    # Windows PowerShell (示例)
    .\project\Scripts\Activate.ps1
    pip install -r requirements.txt
    ```
    主要依赖包括: `langchain`, `langgraph`, `langchain-openai`, `langchain-deepseek`, `faiss-cpu`, `streamlit`, `pydantic`, `python-dotenv`, `langchain-community`, `pypdf`, `dashscope`。

3.  **配置API密钥**
    在项目根目录下，创建一个名为 `.env` 的文件。这个文件用于存放您的私密信息，不会被上传到 GitHub。
    ```
    # .env 文件内容
    DEEPSEEK_API_KEY="sk-..."
    DASHSCOPE_API_KEY="sk-..."
    ```

4.  **准备知识库**
    将您的产品说明书 PDF 文件命名为 `ES.pdf` 并放置在项目根目录下。

## 🏃 如何运行

1.  **首次运行**:
    首次运行时，程序会自动读取 `ES.pdf` 并构建索引，缓存到 `temp/faiss_cache/`；会话记忆与语义缓存保存在 `temp/conversation_store/`。

2.  **执行命令**:
    在项目根目录下打开终端，运行以下命令：
    ```bash
    python -m agent.main
    ```
    程序将启动 Streamlit 页面，在左侧完成模型与知识库配置后即可开始对话。

## ✅ 测试命令

1.  **运行全部测试**
    ```bash
    python -m unittest discover -s tests -p "test_*.py"
    ```

2.  **运行单个测试**
    ```bash
    python -m unittest tests.test_workflow_router.TestRouteAfterAnalysis.test_route_to_default_for_default_quality
    ```

## 🧠 工作流程

Agent 的工作流程被构建为一个状态图 (State Graph):

1.  **入口 (START)**: 接收原始的用户评论字符串。

2.  **分析节点 (`analyze_review`)**:
    - 调用一个配置了结构化输出的 LLM。
    - 将原始评论解析为一个包含 `quality`, `emotion`, `key_information` 的对象。

3.  **条件路由 (`route_after_analysis`)**:
    - 检查分析结果。工具调用判定采用“关键词规则 + LLM判定”的混合策略，优先使用规则命中结果。
    - 当模型将有效提问误判为 `default` 时，会由规则层进行纠偏，避免直接返回“感谢您的评价”。
    - 路由优先级：`tool_use` > `default` > `负面` > `正面/中性`。

4.  **会话持久化**:
    - 图运行时使用 LangGraph checkpointer，并基于 `thread_id` 维持会话上下文，支持同会话连续调用。
    - 调用层会维护线程级会话记忆，并在高相似问题下触发语义缓存命中。

5.  **生成节点 (`generate_*`)**:
    - **`generate_positive_reply` / `generate_default_reply`**: 返回预设的模板化回复。
    - **`generate_negative_reply`**: 启动一个具备工具调用能力的子 Agent。这个 Agent 会根据收到的指令，自主判断是否需要调用 `read_instructions` 工具来查询说明书，然后生成包含解决方案的、安抚性的回复。
    - 对命中高风险关键词（如退款、安全、受伤）的输入，统一返回“转人工优先处理”回复。

6.  **回复校验节点 (`validate_reply`)**:
    - 对生成结果做场景一致性校验（例如使用咨询场景禁止“感谢评价”模板回复）。
    - 校验失败时返回安全兜底回复。

7.  **终点 (END)**: 输出最终生成回复。
