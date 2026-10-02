from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AdvisoryState(TypedDict, total=False):
    # ─── Conversational Message History ──────────────────────────────────────
    # LangGraph's add_messages reducer automatically appends new messages to
    # the existing checkpoint list across turns, giving genuine multi-turn
    # persistence without any manual list-splicing.
    messages: Annotated[List[BaseMessage], add_messages]

    # ─── Session identifier and current user input ───────────────────────────
    session_id: str
    current_input: str

    # ─── Extracted entity state (persisted across session turns) ────────────
    location_name: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    activity: Optional[str]
    timeframe: Optional[str]

    # ─── Weather fetching state ──────────────────────────────────────────────
    weather_data: Optional[Dict[str, Any]]
    weather_error: Optional[str]
    simulate_weather_failure: bool

    # ─── SOP Evaluation state ────────────────────────────────────────────────
    matched_sops: List[Dict[str, Any]]
    primary_sop: Optional[Dict[str, Any]]
    contributing_sops: List[Dict[str, Any]]
    sop_matched: bool

    # ─── Branch routing flags ────────────────────────────────────────────────
    location_resolved: bool
    weather_fetched: bool

    # ─── Final generated output & decision trace ─────────────────────────────
    final_response: str
    response_type: str  # "sop_advisory" | "no_guidance" | "location_unresolved" | "api_error"
    decision_trace: Dict[str, Any]
