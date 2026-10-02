from typing import Literal
from graph.state import AdvisoryState

def route_after_weather_fetch(state: AdvisoryState) -> Literal["handle_location_missing_fallback", "handle_weather_error_fallback", "evaluate_safety_sops"]:
    """
    Branch 1: Verifies whether location was resolved and live weather data was successfully obtained.
    Routes to honest fallbacks if either failed.
    """
    if not state.get("location_resolved", False):
        return "handle_location_missing_fallback"
    
    if not state.get("weather_fetched", False):
        return "handle_weather_error_fallback"
    
    return "evaluate_safety_sops"


def route_after_sop_evaluation(state: AdvisoryState) -> Literal["generate_sop_advisory", "handle_no_guidance_fallback"]:
    """
    Branch 2: Evaluates whether an approved policy covers the situation.
    If no SOP matches, strictly branches to the honest 'no guidance' fallback.
    """
    if state.get("sop_matched", False):
        return "generate_sop_advisory"
    
    return "handle_no_guidance_fallback"
