from typing import Dict, Any
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import AdvisoryState
from graph.nodes import (
    extract_intent_and_entities,
    resolve_and_fetch_weather,
    evaluate_safety_sops,
    generate_sop_advisory,
    handle_no_guidance_fallback,
    handle_location_missing_fallback,
    handle_weather_error_fallback
)
from graph.edges import (
    route_after_weather_fetch,
    route_after_sop_evaluation
)

def build_advisory_graph():
    """
    Constructs the LangGraph state graph with authentic conditional branching,
    safety fallbacks, and durable multi-turn state preservation via MemorySaver.

    Message history is persisted via the add_messages reducer defined in
    AdvisoryState — appended inside nodes, not after graph invocation.
    """
    builder = StateGraph(AdvisoryState)

    # Register Nodes
    builder.add_node("extract_intent_and_entities", extract_intent_and_entities)
    builder.add_node("resolve_and_fetch_weather", resolve_and_fetch_weather)
    builder.add_node("evaluate_safety_sops", evaluate_safety_sops)
    builder.add_node("generate_sop_advisory", generate_sop_advisory)
    builder.add_node("handle_no_guidance_fallback", handle_no_guidance_fallback)
    builder.add_node("handle_location_missing_fallback", handle_location_missing_fallback)
    builder.add_node("handle_weather_error_fallback", handle_weather_error_fallback)

    # Initial flow
    builder.add_edge(START, "extract_intent_and_entities")
    builder.add_edge("extract_intent_and_entities", "resolve_and_fetch_weather")

    # Branch 1: Location & Weather Fetch validation
    builder.add_conditional_edges(
        "resolve_and_fetch_weather",
        route_after_weather_fetch,
        {
            "evaluate_safety_sops": "evaluate_safety_sops",
            "handle_location_missing_fallback": "handle_location_missing_fallback",
            "handle_weather_error_fallback": "handle_weather_error_fallback",
        }
    )

    # Branch 2: SOP Policy Coverage validation
    builder.add_conditional_edges(
        "evaluate_safety_sops",
        route_after_sop_evaluation,
        {
            "generate_sop_advisory": "generate_sop_advisory",
            "handle_no_guidance_fallback": "handle_no_guidance_fallback",
        }
    )

    # Terminal edges
    builder.add_edge("generate_sop_advisory", END)
    builder.add_edge("handle_no_guidance_fallback", END)
    builder.add_edge("handle_location_missing_fallback", END)
    builder.add_edge("handle_weather_error_fallback", END)

    # MemorySaver checkpointer — messages accumulate via add_messages reducer
    # across turns within the same thread_id session.
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory)
    return graph

# Compiled singleton graph
graph_app = build_advisory_graph()

def run_advisory_agent(
    session_id: str,
    user_input: str,
    simulate_weather_failure: bool = False
) -> Dict[str, Any]:
    """
    Executes one turn of the Weather Advisory Bot within a conversational session thread.
    Location, activity, timeframe, and full message history are persisted across turns
    via LangGraph's MemorySaver checkpointer and the add_messages reducer in AdvisoryState.
    """
    config = {"configurable": {"thread_id": session_id}}

    # Messages are appended inside nodes (extract_intent → HumanMessage,
    # response terminal nodes → AIMessage), so the LangGraph checkpoint
    # captures them. We do NOT mutate them after invoke().
    initial_payload = {
        "session_id": session_id,
        "current_input": user_input,
        "simulate_weather_failure": simulate_weather_failure,
    }

    result = graph_app.invoke(initial_payload, config=config)
    return result
