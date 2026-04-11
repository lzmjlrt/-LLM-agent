from langchain_core.messages import HumanMessage
import logging

from agent.workflow.schema import AgentState

logger = logging.getLogger(__name__)


def create_nodes(structred_llm, agent_executor):
    """创建工作流节点。"""

    def analyze_review(state: AgentState) -> dict:
        request_id = state.get("request_id", "n/a")
        logger.info("--- [节点执行] 正在分析评论 --- request_id=%s", request_id)
        review_analysis = structred_llm.invoke(state["original_review"])
        return {"review_quality": review_analysis}

    def generate_negative_reply(state: AgentState) -> dict:
        request_id = state.get("request_id", "n/a")
        logger.info("--- [节点执行] 正在处理负面评论 --- request_id=%s", request_id)
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
        request_id = state.get("request_id", "n/a")
        logger.info("--- [节点执行] 正在处理正面/中性评论 --- request_id=%s", request_id)
        return {"finally_reply": "亲，非常感谢您的认可与支持！您的满意是我们不断前行的动力，期待您的再次光临！"}

    def generate_default_reply(state: AgentState) -> dict:
        request_id = state.get("request_id", "n/a")
        logger.info("--- [节点执行] 正在处理无效评论 --- request_id=%s", request_id)
        return {"finally_reply": "感谢您的评价！"}

    return {
        "analyze_review": analyze_review,
        "generate_negative_reply": generate_negative_reply,
        "generate_positive_reply": generate_positive_reply,
        "generate_default_reply": generate_default_reply,
    }
