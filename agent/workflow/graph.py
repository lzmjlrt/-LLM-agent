from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from agent.workflow.nodes import create_nodes
from agent.workflow.router import route_after_analysis
from agent.workflow.schema import AgentState, ReviewQuality


def create_graph(llm, tools, checkpointer=None):
    """根据传入的 LLM 和工具构建并编译 LangGraph。"""
    structred_llm = llm.with_structured_output(ReviewQuality, strict=True)
    agent_executor = create_react_agent(llm, tools)
    nodes = create_nodes(structred_llm, agent_executor)

    workflow = StateGraph(AgentState)
    workflow.add_node("analyze_review", nodes["analyze_review"])
    workflow.add_node("generate_negative_reply", nodes["generate_negative_reply"])
    workflow.add_node("generate_positive_reply", nodes["generate_positive_reply"])
    workflow.add_node("generate_default_reply", nodes["generate_default_reply"])
    workflow.add_node("validate_reply", nodes["validate_reply"])

    workflow.add_edge(START, "analyze_review")
    workflow.add_conditional_edges(
        "analyze_review",
        route_after_analysis,
        {
            "generate_negative_reply": "generate_negative_reply",
            "generate_positive_reply": "generate_positive_reply",
            "generate_default_reply": "generate_default_reply",
        },
    )
    workflow.add_edge("generate_negative_reply", "validate_reply")
    workflow.add_edge("generate_positive_reply", "validate_reply")
    workflow.add_edge("generate_default_reply", "validate_reply")
    workflow.add_edge("validate_reply", END)

    return workflow.compile(checkpointer=checkpointer)
