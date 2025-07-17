# 智能评论回复 Agent

这是一个基于 LangChain 和 LangGraph 构建的智能客服项目，旨在自动分析电商平台上的用户评论，并根据评论内容生成高度定制化的回复。

## ✨ 项目特色

- **智能分诊**: 自动分析评论的情感（正面/负面/中性）和有效性（真实评论/默认好评）。
- **动态路由**: 使用 LangGraph 的条件路由，将不同类型的评论分发给不同的处理逻辑。
- **RAG 增强**: 对于提及“产品不会用”的负面评论，能自动从本地的 PDF 说明书 (`ES.pdf`) 中检索相关信息，并整合到回复中。
- **工具调用**: 在负面评论处理流程中，Agent 被授权可以自主决定是否调用 `read_instructions` 工具来查询知识库。
- **模块化设计**: 代码结构清晰，分为配置、RAG 设置、图工作流和主入口，易于维护和扩展。

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
│   ├── config.py           # 配置文件 (API密钥, 路径, 模型名称)
│   ├── rag_setup.py        # RAG设置 (加载PDF, 创建向量库, 定义工具)
│   ├── graph_workflow.py   # 核心工作流 (定义状态, 节点, 构建图)
│   └── main.py             # 项目入口
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
    pip install -r requirements.txt
    ```
    *如果 `requirements.txt` 文件不存在，您可以通过 `pip freeze > requirements.txt` 命令生成。*
    主要依赖包括: `langchain`, `langgraph`, `faiss-cpu`, `pydantic`, `python-dotenv`, `langchain-community`, `pypdf`, `dashscope`。

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
    首次运行时，程序会自动读取 `ES.pdf` 文件，创建向量索引并保存在 `emb/` 文件夹下。这个过程可能需要一些时间。

2.  **执行命令**:
    在项目根目录下打开终端，运行以下命令：
    ```bash
    python agent/main.py
    ```
    程序将执行 `main.py` 中预设的几个测试用例，并打印出 Agent 的完整思考流程和最终回复。

## 🧠 工作流程

Agent 的工作流程被构建为一个状态图 (State Graph):

1.  **入口 (START)**: 接收原始的用户评论字符串。

2.  **分析节点 (`analyze_review`)**:
    - 调用一个配置了结构化输出的 LLM。
    - 将原始评论解析为一个包含 `quality`, `emotion`, `key_information` 的对象。

3.  **条件路由 (`route_after_analysis`)**:
    - 检查分析结果。
    - **如果** 评论是无效的 (`default`) -> 跳转到 `generate_default_reply`。
    - **如果** 评论是负面的 (`负面`) -> 跳转到 `generate_negative_reply`。
    - **否则** (正面/中性) -> 跳转到 `generate_positive_reply`。

4.  **生成节点**:
    - **`generate_positive_reply` / `generate_default_reply`**: 返回预设的模板化回复。
    - **`generate_negative_reply`**: 启动一个具备工具调用能力的子 Agent。这个 Agent 会根据收到的指令，自主判断是否需要调用 `read_instructions` 工具来查询说明书，然后生成包含解决方案的、安抚性的回复。

5.  **终点 (END)**: 输出最终生成的回复内
