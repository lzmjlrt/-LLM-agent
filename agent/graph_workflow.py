from typing import List, TypedDict
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

#from agent import config
# from agent.rag_setup import tools


# --- LLM 和 Agent Executor 初始化 ---
# llm = init_chat_model(
#     config.LLM_MODEL_NAME,
#     model_provider="deepseek",
#     base_url=config.DEEPSEEK_BASE_URL,
#     api_key=config.DEEPSEEK_API_KEY
# )

# 1. 结构化输出模型
class ReviewQuality(BaseModel):
    """评价质量分类器的输出要求"""
    quality: str = Field(description="是否是正常评论，或者是默认的回复内容，返回值为normal或者default")
    emotion: str = Field(description="情感倾向。只返回正面，负面，中性这三个词")
    key_information: List[str] = Field(description="提取评价里用户对产品不会用的关键信息，返回一个列表，列表里是用户对产品的关键信息")
    require_tool_use: bool = Field(description="判读用户的评论是否需要调用工具来获取更多信息，True表示需要，False表示不需要")
#structred_llm = llm.with_structured_output(ReviewQuality)

# 2. 专门用于负面评论的工具调用 Agent
#agent_executor = create_react_agent(llm.bind_tools(tools), tools)

# --- LangGraph 状态和节点定义 ---
class AgentState(TypedDict):
    original_review: str
    review_quality: ReviewQuality
    finally_reply: str

def create_graph(llm, tools):
    """根据传入的LLM和工具构建并编译 LangGraph"""
    
    # 1. 动态创建结构化LLM和Agent Executor
    structred_llm = llm.with_structured_output(ReviewQuality)
    agent_executor = create_react_agent(llm, tools)

    # --- 节点定义 (现在它们可以访问外部作用域的 structred_llm 和 agent_executor) ---
    def analyze_review(state: AgentState) -> dict:
        """节点1: 分析评论"""
        print("--- [节点执行] 正在分析评论 ---")
        review_analysis = structred_llm.invoke(state["original_review"])
        return {"review_quality": review_analysis}

    def generate_negative_reply(state: AgentState) -> dict:
        """节点2: 为负面评论生成回复 (会调用工具)"""
        print("--- [节点执行] 正在处理负面评论 ---")
        prompt_content = (
            f"你是一个专业的客服。顾客给出了负面评价：'{state['original_review']}'。\n"
            "你的任务是生成一个真诚、有同理心且专业的回复。\n"
            "1. 首先，分析用户的评价。如果用户提到了具体产品不会用的问题（例如：不知道如何更换内刀头），你**必须**使用 `read_instructions` 工具来查找解决方案。\n"
            "2. 然后，将工具返回的信息整合到你的最终回复中，先表示歉意，再提供清晰的步骤或解决方案。\n"
            "3. 如果用户只是抱怨其他问题（如质量、物流），则无需调用工具，直接生成安抚性的回复。\n"
            "4. 如果你调用工具检索到的信息和用户问题无关，请直接生成安抚性的回复。\n"
        )
        response = agent_executor.invoke({"messages": [HumanMessage(content=prompt_content)]})
        return {"finally_reply": response["messages"][-1].content}

    def generate_positive_reply(state: AgentState) -> dict:
        """节点3: 为正面/中性评论生成回复"""
        print("--- [节点执行] 正在处理正面/中性评论 ---")
        reply_content = "亲，非常感谢您的认可与支持！您的满意是我们不断前行的动力，期待您的再次光临！"
        return {"finally_reply": reply_content}

    def generate_default_reply(state: AgentState) -> dict:
        """节点4: 为无效评论生成通用回复"""
        print("--- [节点执行] 正在处理无效评论 ---")
        reply_content = "感谢您的评价！"
        return {"finally_reply": reply_content}

    # --- 路由逻辑 (保持不变) ---
    def route_after_analysis(state: AgentState) -> str:
        """决策路由"""
        print("--- [决策] 正在根据分析结果进行路由... ---")
        analysis_result = state["review_quality"]
        if analysis_result.require_tool_use:
            print("--- [决策结果] -> 需要调用工具 (路由至负面评论处理节点) ---")
            return "generate_negative_reply"
        if analysis_result.quality == "default":
            print("--- [决策结果] -> 无效评论 ---")
            return "generate_default_reply"
        if analysis_result.emotion == "负面":
            print("--- [决策结果] -> 负面评论 ---")
            return "generate_negative_reply"
        else:
            print("--- [决策结果] -> 正面/中性评论 ---")
            return "generate_positive_reply"

    # --- 构建图 (保持不变) ---
    workflow = StateGraph(AgentState)
    
    workflow.add_node("analyze_review", analyze_review)
    workflow.add_node("generate_negative_reply", generate_negative_reply)
    workflow.add_node("generate_positive_reply", generate_positive_reply)
    workflow.add_node("generate_default_reply", generate_default_reply)
    
    workflow.add_edge(START, "analyze_review")
    
    workflow.add_conditional_edges(
        "analyze_review",
        route_after_analysis,
        {
            "generate_negative_reply": "generate_negative_reply",
            "generate_positive_reply": "generate_positive_reply",
            "generate_default_reply": "generate_default_reply"
        }
    )
    
    workflow.add_edge("generate_negative_reply", END)
    workflow.add_edge("generate_positive_reply", END)
    workflow.add_edge("generate_default_reply", END)
    
    return workflow.compile()