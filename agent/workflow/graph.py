from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

from agent.workflow.constants import (
    ROUTE_GENERATE_DEFAULT_REPLY,
    ROUTE_GENERATE_NEGATIVE_REPLY,
    ROUTE_GENERATE_POSITIVE_REPLY,
)
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
    workflow.add_node(ROUTE_GENERATE_NEGATIVE_REPLY, nodes[ROUTE_GENERATE_NEGATIVE_REPLY])
    workflow.add_node(ROUTE_GENERATE_POSITIVE_REPLY, nodes[ROUTE_GENERATE_POSITIVE_REPLY])
    workflow.add_node(ROUTE_GENERATE_DEFAULT_REPLY, nodes[ROUTE_GENERATE_DEFAULT_REPLY])
    workflow.add_node("validate_reply", nodes["validate_reply"])

    workflow.add_edge(START, "analyze_review")
    workflow.add_conditional_edges(
        "analyze_review",
        route_after_analysis,
        {
            ROUTE_GENERATE_NEGATIVE_REPLY: ROUTE_GENERATE_NEGATIVE_REPLY,
            ROUTE_GENERATE_POSITIVE_REPLY: ROUTE_GENERATE_POSITIVE_REPLY,
            ROUTE_GENERATE_DEFAULT_REPLY: ROUTE_GENERATE_DEFAULT_REPLY,
        },
    )
    workflow.add_edge(ROUTE_GENERATE_NEGATIVE_REPLY, "validate_reply")
    workflow.add_edge(ROUTE_GENERATE_POSITIVE_REPLY, "validate_reply")
    workflow.add_edge(ROUTE_GENERATE_DEFAULT_REPLY, "validate_reply")
    workflow.add_edge("validate_reply", END)

    return workflow.compile(checkpointer=checkpointer)
