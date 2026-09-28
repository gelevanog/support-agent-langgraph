"""Assembles the LangGraph state machine."""

from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from support_agent.graph.nodes import AgentDeps, SupportAgentNodes
from support_agent.graph.state import AgentState

SupportGraph = CompiledStateGraph[AgentState, None, AgentState, AgentState]


def build_graph(deps: AgentDeps, checkpointer: BaseCheckpointSaver[str] | None = None) -> SupportGraph:
    """
    classify -> research <-> lookup_tools -> apply_policy -+-> execute_action -+-> draft_reply -> END
                                                           +-> human_approval -+        ^
                                                           +-> escalate ----------------+
                                                           +-> draft_reply (deny / inform / ask)
    """
    nodes = SupportAgentNodes(deps)
    graph = StateGraph(AgentState)

    graph.add_node("classify", nodes.classify)
    graph.add_node("research", nodes.research)
    graph.add_node("lookup_tools", nodes.lookup_tools)
    graph.add_node("apply_policy", nodes.apply_policy)
    graph.add_node("human_approval", nodes.human_approval)
    graph.add_node("execute_action", nodes.execute_action)
    graph.add_node("escalate", nodes.escalate)
    graph.add_node("draft_reply", nodes.draft_reply)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "research")
    graph.add_conditional_edges(
        "research",
        nodes.route_after_research,
        {"lookup_tools": "lookup_tools", "apply_policy": "apply_policy"},
    )
    graph.add_edge("lookup_tools", "research")
    graph.add_conditional_edges(
        "apply_policy",
        nodes.route_by_verdict,
        {
            "human_approval": "human_approval",
            "execute_action": "execute_action",
            "escalate": "escalate",
            "draft_reply": "draft_reply",
        },
    )
    graph.add_conditional_edges(
        "human_approval",
        nodes.route_after_approval,
        {"execute_action": "execute_action", "draft_reply": "draft_reply"},
    )
    graph.add_conditional_edges(
        "execute_action",
        nodes.route_after_action,
        {"escalate": "escalate", "draft_reply": "draft_reply"},
    )
    graph.add_edge("escalate", "draft_reply")
    graph.add_edge("draft_reply", END)

    return graph.compile(checkpointer=checkpointer)
