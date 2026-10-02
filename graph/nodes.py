import re
import logging
from typing import Dict, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage

from graph.state import AdvisoryState
from services.weather_service import WeatherService
from services.llm_service import LLMService
from policies.sop_engine import SOPEngine

logger = logging.getLogger(__name__)

weather_service = WeatherService()
llm_service = LLMService()
sop_engine = SOPEngine()

# Common cities pattern recognizer for robust zero-dependency extraction
COMMON_LOCATIONS = [
    "bhopal", "mumbai", "delhi", "bangalore", "bengaluru", "chennai", "kolkata",
    "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow", "chandigarh", "goa",
    "springfield", "london", "new york", "tokyo", "paris", "berlin", "san francisco",
    "sydney", "singapore", "madhya pradesh", "tamil nadu", "indore", "patna"
]

def extract_intent_and_entities(state: AdvisoryState) -> Dict[str, Any]:
    """
    Parses user query and conversational history to resolve:
    - Activity (cycling, running, picnic, driving, pet walking, etc.)
    - Location (e.g. Bhopal, Dubai, Chicago, Phoenix, London, Paris)
    - Timeframe (today, this evening, afternoon, tomorrow)
    Preserves existing session context when follow-ups are asked.
    """
    user_input = state.get("current_input", "") or ""
    lower_input = user_input.lower()

    # 1. Location Extraction
    extracted_location = None
    stopwords = {
        "the", "a", "an", "my", "work", "school", "park", "office", "home",
        "general", "some", "this", "today", "terms", "developer", "now",
        "outdoor", "open", "black", "road", "roads", "street", "streets", "fast"
    }
    
    # Preposition candidates: 'in <city>', 'at <city>', 'near <city>', 'around <city>'
    matches = re.findall(r'\b(?:in|at|near|around)\s+([A-Za-z]+)', user_input)
    valid_candidates = [m for m in matches if m.lower() not in stopwords]
    if valid_candidates:
        extracted_location = valid_candidates[-1].strip().title()

    if not extracted_location:
        for loc in COMMON_LOCATIONS:
            if re.search(rf'\b{re.escape(loc)}\b', lower_input):
                extracted_location = loc.title()
                break

    # Direct short location input (e.g. user answering location prompt with "Mainpuri" or "New Delhi")
    if not extracted_location:
        cleaned_input = re.sub(r'^(?:it is|it\'s|its|city is|in|at|for|near)\s+', '', user_input.strip(), flags=re.I).strip('.?!, ')
        words = cleaned_input.split()
        is_short_location = (
            1 <= len(words) <= 3 and
            not any(w.lower() in ["is", "can", "should", "what", "how", "why", "who", "where", "today", "tomorrow", "tonight", "safe", "cycling", "running", "picnic", "drive", "travel", "yes", "no"] for w in words)
        )
        if is_short_location and cleaned_input.lower() not in stopwords:
            extracted_location = cleaned_input.title()

    is_new_location = bool(extracted_location)
    final_location = extracted_location or state.get("location_name")

    # 2. Activity Extraction (Semantic paraphrases supported)
    extracted_activity = None
    activity_keywords = {
        "cycling": [
            "cycle", "cycling", "bike", "biking", "bicycle", "two-wheeler",
            "two-wheeled", "pedal machine", "scooter", "motorcycle", "pedal"
        ],
        "running": [
            "run", "running", "jog", "jogging", "cardio", "sprint", "workout"
        ],
        "picnic": [
            "picnic", "park", "bbq", "barbecue", "outing", "hangout", "lawn party"
        ],
        "commute": [
            "drive", "driving", "travel", "commute", "highway", "road trip", "car ride"
        ],
        "pet walking": [
            "dog walk", "walk the dog", "dog", "puppy", "pup", "pet", "retriever",
            "canine", "four-legged"
        ],
        "general outdoor": [
            "outside", "outdoor", "walk", "stroll", "play"
        ]
    }

    for act_name, kw_list in activity_keywords.items():
        if any(re.search(rf'\b{re.escape(kw)}\b', lower_input) for kw in kw_list):
            extracted_activity = act_name
            break

    final_activity = extracted_activity or state.get("activity") or "general outdoor"

    # 3. Timeframe Extraction
    extracted_timeframe = "current"
    if "this evening" in lower_input or "tonight" in lower_input or "evening" in lower_input:
        extracted_timeframe = "this evening"
    elif "afternoon" in lower_input or "midday" in lower_input or "1:00 pm" in lower_input:
        extracted_timeframe = "afternoon"
    elif "tomorrow" in lower_input:
        extracted_timeframe = "tomorrow"
    elif "today" in lower_input:
        extracted_timeframe = "today"
    elif state.get("timeframe"):
        extracted_timeframe = state.get("timeframe")

    logger.info(f"Entities parsed: Location='{final_location}', Activity='{final_activity}', Timeframe='{extracted_timeframe}'")

    updates = {
        "activity": final_activity,
        "timeframe": extracted_timeframe,
    }
    if is_new_location:
        updates["location_name"] = final_location
        updates["latitude"] = None
        updates["longitude"] = None
    elif final_location:
        updates["location_name"] = final_location

    return updates


def resolve_and_fetch_weather(state: AdvisoryState) -> Dict[str, Any]:
    """
    Geocodes city and pulls live telemetry from Open-Meteo.
    Handles network errors, unresolved locations, and simulated failure tests.
    Reuses coordinates across session turns if location has not changed.
    """
    location = state.get("location_name")
    timeframe = state.get("timeframe")
    simulate_failure = state.get("simulate_weather_failure", False)
    lat = state.get("latitude")
    lon = state.get("longitude")

    if not location:
        return {
            "location_resolved": False,
            "weather_fetched": False,
            "weather_error": "Location was not specified."
        }

    # If coordinates are already present in session state and no failure simulated, reuse directly
    if lat is not None and lon is not None and not simulate_failure:
        weather_res = weather_service.fetch_weather(
            latitude=lat,
            longitude=lon,
            timeframe=timeframe,
            simulate_failure=simulate_failure
        )
        if weather_res.get("success"):
            return {
                "weather_data": weather_res,
                "weather_error": None,
                "location_resolved": True,
                "weather_fetched": True,
            }

    # Otherwise clean city name (take primary part before commas)
    search_city = location.split(",")[0].strip()

    weather_result = weather_service.get_weather_for_city(
        city_name=search_city,
        timeframe=timeframe,
        simulate_failure=simulate_failure
    )

    if not weather_result.get("success"):
        stage = weather_result.get("stage", "unknown")
        err_msg = weather_result.get("error", "Unknown weather service error")
        return {
            "location_resolved": (stage != "geocoding"),
            "weather_fetched": False,
            "weather_error": err_msg,
            "weather_data": None
        }

    loc_info = weather_result.get("location", {})
    weather_data = weather_result.get("weather", {})
    formatted_name = f"{loc_info.get('city')}, {loc_info.get('admin1', '')} {loc_info.get('country', '')}".strip(" ,")

    return {
        "location_name": formatted_name,
        "latitude": loc_info.get("latitude"),
        "longitude": loc_info.get("longitude"),
        "weather_data": weather_data,
        "weather_error": None,
        "location_resolved": True,
        "weather_fetched": True,
    }


def evaluate_safety_sops(state: AdvisoryState) -> Dict[str, Any]:
    """
    Executes the decoupled SOP evaluation engine.
    Checks live weather numbers against policy thresholds and fuzzy envelopes.
    Ranks matching SOPs by severity rank descending.
    """
    weather = state.get("weather_data", {})
    activity = state.get("activity", "general")
    query = state.get("current_input", "")

    eval_result = sop_engine.evaluate(weather=weather, activity=activity, query_text=query)

    is_matched = eval_result.get("matched", False)
    primary = eval_result.get("primary_sop")
    contributing = eval_result.get("contributing_sops", [])
    all_matches = eval_result.get("all_matches", [])

    return {
        "sop_matched": is_matched,
        "primary_sop": primary,
        "contributing_sops": contributing,
        "matched_sops": all_matches,
        "decision_trace": {
            "evaluated_activity": activity,
            "matched_count": len(all_matches),
            "primary_id": primary.get("id") if primary else None,
            "primary_severity": primary.get("severity") if primary else None,
        }
    }


def generate_sop_advisory(state: AdvisoryState) -> Dict[str, Any]:
    """
    Invokes LLM service under strict prompt constraints to compose
    an advisory grounded in the verified weather telemetry and primary SOP.
    """
    primary = state.get("primary_sop", {})
    contributing = state.get("contributing_sops", [])
    weather = state.get("weather_data", {})
    location = state.get("location_name", "Unknown Location")
    query = state.get("current_input", "")

    # Context from conversation
    history_messages = state.get("messages", [])
    context_str = ""
    if len(history_messages) > 1:
        recent = history_messages[-3:-1]
        context_str = " | ".join([f"{m.type}: {m.content[:80]}" for m in recent])

    response_text = llm_service.compose_sop_response(
        query=query,
        location_name=location,
        weather=weather,
        primary_sop=primary,
        contributing_sops=contributing,
        conversation_context=context_str
    )

    return {
        "final_response": response_text,
        "response_type": "sop_advisory",
    }


def handle_no_guidance_fallback(state: AdvisoryState) -> Dict[str, Any]:
    """
    Honest refusal when no written SOP applies to the scenario or weather condition.
    Prevents hallucinating advice.
    """
    activity = state.get("activity", "outdoor activity")
    location = state.get("location_name", "your area")
    weather = state.get("weather_data", {})

    temp = weather.get("temperature_2m", "N/A")
    precip = weather.get("precipitation", "N/A")
    wind = weather.get("wind_speed_10m", "N/A")

    msg = (
        f"**Notice: No Policy Guidance Available**\n\n"
        f"We do not currently have an approved Standard Operating Procedure (SOP) covering **'{activity}'** "
        f"under the current conditions observed for **{location}** "
        f"(Temperature: {temp}°C, Precipitation: {precip} mm, Wind: {wind} km/h).\n\n"
        f"Because our service stands behind verified safety guidance, our assistant is strictly prohibited "
        f"from improvising or guessing advice. For your personal safety, please refer to local municipal "
        f"advisories or specialized regional sports guidelines."
    )

    return {
        "final_response": msg,
        "response_type": "no_guidance"
    }


def handle_location_missing_fallback(state: AdvisoryState) -> Dict[str, Any]:
    """
    Prompts the user for location when it cannot be inferred from query or history.
    """
    activity = state.get("activity", "your outdoor plans")
    msg = (
        f"Could you please specify which **city or location** you are asking about? "
        f"To evaluate our safety Standard Operating Procedures (SOPs) for {activity}, "
        f"I need to check verified live meteorological data from Open-Meteo for that specific area."
    )
    return {
        "final_response": msg,
        "response_type": "location_unresolved"
    }


def handle_weather_error_fallback(state: AdvisoryState) -> Dict[str, Any]:
    """
    Honest failure when geocoding fails or meteorological endpoints are unreachable.
    """
    location = state.get("location_name", "the requested location")
    error = state.get("weather_error", "Weather service unreachable")

    msg = (
        f"**Service Notice: Meteorological Data Unavailable**\n\n"
        f"We were unable to pull live weather observations for **{location}**.\n"
        f"*Reason:* `{error}`\n\n"
        f"Under our safety compliance policy, our assistant will never guess or extrapolate weather conditions. "
        f"Please verify the location name or retry in a few moments."
    )
    return {
        "final_response": msg,
        "response_type": "api_error"
    }
