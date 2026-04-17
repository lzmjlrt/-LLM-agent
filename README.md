# 智能电商客服 Agent（LangChain + LangGraph）

面向“商品评价/使用咨询/售后风险”的可控式 Agent。  
核心目标不是“能答”，而是“**可路由、可检索、可校验、可追踪**”。

## ✨ 项目特色

- **Graph-based Orchestration**：`analyze -> route -> generate -> validate`
- **Structured Output Contract**：`ReviewQuality` 约束路由输入（quality/emotion/require_tool_use）
- **Hybrid Decisioning**：规则优先 + LLM 回退（降低路由抖动）
- **Agentic RAG**：`rewritten_query` 首检 -> LLM 相关性判定 -> Multi Query -> 查询分解（按需升级）
- **Guardrails**：补充信息分支、超范围分支、高风险转人工分支
- **Post-generation Validation**：生成后场景一致性校验，拦截“感谢评价”类误答
- **Thread-level Memory + Semantic Cache**：会话摘要与相似问题缓存命中
- **Observability**：结构化事件日志 + `request_id/thread_id` 全链路

> 当前 **未启用 HyDE**（按业务要求暂不引入）。



## 🛠️ 技术栈

- **核心框架**: LangChain, LangGraph
- **大语言模型 (LLM)**: DeepSeek API
- **嵌入模型 (Embedding)**: 阿里巴巴通义千问 (DashScope)
- **向量数据库**: FAISS (本地)
- **数据模型**: Pydantic
- **环境管理**: python-dotenv

---


## 示意图
<img width="1584" height="854" alt="ca6afb01af6329c6a7f322672135b3b0" src="https://github.com/user-attachments/assets/2d864ea6-29e6-4601-841b-e3d544e5d2f1" />

## 1. 运行前提

- Python 3.10+
- Windows / macOS / Linux（本文命令以 Windows PowerShell 为例）

安装依赖：

```powershell
.\project\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 2. 启动方式（推荐）

```powershell
python -m agent.main
```

启动后会打开 Streamlit 页面，配置都在左侧 Sidebar 完成。

---

## 3. 配置方式（重要：不是本地硬编码）

### 3.1 API Key

默认流程是**在页面里输入**：
- LLM Provider: `DeepSeek` / `OpenAI`
- Embedding Provider: `DashScope (Alibaba)` / `OpenAI`



### 3.2 PDF 知识库

默认流程是**在页面里上传 PDF**，然后点击“应用配置”触发初始化。  


---

## 4. RAG 索引与缓存生命周期

当你点击“应用配置”后：

1. 上传文件落盘：`temp/uploads/<thread_id>/`
2. 计算 PDF hash + embedding 配置
3. 生成确定性 FAISS 缓存路径：`temp/faiss_cache/...`
4. 命中缓存则增量复用；未命中则构建新索引
5. 会话记忆与语义缓存写入：`temp/conversation_store/`

---

## 5. Agentic RAG 检索策略（当前实现）

`read_instructions` 工具内部执行以下分层策略：

1. **Stage 1（主查询）**：先用 `原查询 + 规则改写` 检索  
2. **LLM Judge**：判断 `sufficient / insufficient / out_of_scope`  
3. **Stage 2（Multi Query）**：仅在 `insufficient` 时升级  
4. **Stage 3（Decomposition）**：Multi Query 仍不足时再升级  
5. **Boundary Handling**：  
   - `out_of_scope` -> 明确超范围提示  
   - 多轮后仍不足 -> 提示补充型号/现象/步骤

---

## 6. 工作流路由规则

`route_after_analysis` 优先级固定为：

1. `require_tool_use == True`
2. `quality == default`
3. `emotion == 负面`
4. 其他 -> 正面/中性

并保留高风险关键词与“我要转人工”等显式诉求的人工接管逻辑。

---

## 7. 目录结构（当前真实结构）

```text
.
├── agent/
│   ├── app/
│   │   └── chat_app.py
│   ├── factories/
│   │   ├── config_validation.py
│   │   └── model_factory.py
│   ├── rag/
│   │   ├── cache.py
│   │   ├── embeddings.py
│   │   ├── tools.py
│   │   └── vector_store.py
│   ├── services/
│   │   ├── chat_service.py            # facade
│   │   ├── conversation_memory.py
│   │   ├── invoke_service.py
│   │   ├── runtime_initializer.py
│   │   └── upload_store.py
│   ├── workflow/
│   │   ├── constants.py
│   │   ├── graph.py
│   │   ├── intent_rules.py
│   │   ├── node_factory.py
│   │   ├── nodes.py                   # compatibility exports
│   │   ├── reply_policies.py
│   │   ├── router.py
│   │   └── schema.py
│   ├── config.py
│   ├── event_names.py
│   ├── graph_workflow.py              # compatibility exports
│   ├── main.py
│   └── rag_setup.py                   # compatibility exports
├── tests/
├── requirements.txt
└── README.md
```

---

## 8. 测试命令

全量：

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

单测示例：

```powershell
python -m unittest tests.test_workflow_router.TestRouteAfterAnalysis.test_route_to_default_for_default_quality
```

---

## 9. 可选 `.env`（仅高级用法）

项目仍支持从环境变量读取默认配置（如模型名、日志级别、缓存参数），但**不是必需**。  
常用可选项：

- `LOG_LEVEL`
- `CONVERSATION_HISTORY_LIMIT`
- `SEMANTIC_CACHE_ENABLED`
- `SEMANTIC_CACHE_SIMILARITY_THRESHOLD`

---


