import json
import os
from typing import List, Dict, Any, Optional

SOPS_FILE_PATH = os.path.join(os.path.dirname(__file__), "sops.json")

class SOPEngine:
    """
    Decoupled SOP Evaluation Engine.
    Reads policy definitions from sops.json and evaluates them against
    live meteorological metrics and user intent.
    
    Zero code modification is required to add or update SOP rules.
    """
    def __init__(self, sops_path: Optional[str] = None):
        self.sops_path = sops_path or SOPS_FILE_PATH
        self._cached_mtime = 0
        self._sops: List[Dict[str, Any]] = []
        self._load_sops()

    def _load_sops(self) -> List[Dict[str, Any]]:
        """Dynamically reloads SOPs if the underlying JSON file was modified."""
        if not os.path.exists(self.sops_path):
            return []
        
        current_mtime = os.path.getmtime(self.sops_path)
        if current_mtime != self._cached_mtime or not self._sops:
            with open(self.sops_path, "r", encoding="utf-8") as f:
                self._sops = json.load(f)
            self._cached_mtime = current_mtime
        return self._sops

    def get_all_sops(self) -> List[Dict[str, Any]]:
        return self._load_sops()

    def evaluate(
        self,
        weather: Dict[str, Any],
        activity: str,
        query_text: str = ""
    ) -> Dict[str, Any]:
        """
        Evaluates active SOPs against current live weather metrics and user activity.
        
        Resolution Strategy:
        1. Identifies all candidate SOPs whose applicable_activities match the user's intent.
        2. Evaluates the conditions (threshold, composite fuzzy, or extreme weather system).
        3. Sorts all passing SOPs by severity_rank descending (CRITICAL -> WARNING -> CAUTION -> ADVISORY).
        4. Designates the top match as primary_sop, and secondary matches as contributing_sops.
        """
        sops = self._load_sops()
        matched_sops = []

        norm_activity = (activity or "general").lower().strip()
        norm_query = (query_text or "").lower()

        # Telemetry metrics from Open-Meteo
        temp = float(weather.get("temperature_2m", 20.0))
        apparent_temp = float(weather.get("apparent_temperature", temp))
        humidity = float(weather.get("relative_humidity_2m", 50.0))
        precip = float(weather.get("precipitation", 0.0))
        rain = float(weather.get("rain", precip))
        precip_prob = float(weather.get("precipitation_probability", 0.0))
        wind_speed = float(weather.get("wind_speed_10m", 0.0))
        wind_gusts = float(weather.get("wind_gusts_10m", wind_speed))
        uv_index = float(weather.get("uv_index", 0.0))
        weather_code = int(weather.get("weather_code", 0))

        # Check cyclonic or low-pressure keywords in query or high monsoon indicators
        # (e.g. sustained heavy rain > 15mm or probability > 80% + high gusts)
        is_monsoon_low_pressure = (
            precip >= 15.0 or 
            (precip_prob >= 80 and precip >= 5.0) or
            (wind_speed >= 45.0) or
            (wind_gusts >= 55.0) or
            any(k in norm_query for k in ["low pressure", "depression", "cyclonic", "monsoon", "imd", "bhopal", "squally"])
            and (precip >= 5.0 or wind_speed >= 35.0)
        )

        for sop in sops:
            applies_to_activity = False
            tags = [t.lower() for t in sop.get("applicable_activities", [])]
            if "all" in tags:
                applies_to_activity = True
            elif any(t in norm_activity for t in tags) or any(t in norm_query for t in tags):
                applies_to_activity = True

            if not applies_to_activity:
                continue

            triggers = sop.get("triggers", {})
            condition_met = False
            reasons = []

            # 1. Extreme system / SOP-001 check
            if sop.get("id") == "SOP-001":
                if is_monsoon_low_pressure or precip >= triggers.get("precipitation_mm_gte", 15.0) or wind_speed >= triggers.get("wind_speed_kmh_gte", 45.0):
                    condition_met = True
                    reasons.append(f"Monsoon low-pressure / heavy precipitation triggers met (Precip: {precip}mm, Wind: {wind_speed}km/h, Gusts: {wind_gusts}km/h).")

            # 2. Thunderstorm check (SOP-002)
            elif sop.get("id") == "SOP-002":
                if (weather_code in triggers.get("weather_codes", [])) or (precip >= triggers.get("precipitation_mm_gte", 10.0) and wind_gusts >= triggers.get("wind_gusts_kmh_gte", 45.0)):
                    condition_met = True
                    reasons.append(f"Thunderstorm/convective criteria met (WMO Code: {weather_code}, Precip: {precip}mm, Gusts: {wind_gusts}km/h).")

            # 3. Numeric threshold evaluations
            elif sop.get("condition_type") == "numeric":
                rule_pass = True
                
                if "wind_speed_kmh_gte" in triggers and wind_speed < triggers["wind_speed_kmh_gte"]:
                    rule_pass = False
                elif "wind_speed_kmh_gte" in triggers:
                    reasons.append(f"Wind speed {wind_speed} km/h >= {triggers['wind_speed_kmh_gte']} km/h")

                if "wind_gusts_kmh_gte" in triggers and wind_gusts < triggers["wind_gusts_kmh_gte"]:
                    rule_pass = False

                if "apparent_temperature_c_gte" in triggers and apparent_temp < triggers["apparent_temperature_c_gte"]:
                    rule_pass = False
                elif "apparent_temperature_c_gte" in triggers:
                    reasons.append(f"Apparent temp {apparent_temp}°C >= {triggers['apparent_temperature_c_gte']}°C")

                if "apparent_temperature_c_lte" in triggers and apparent_temp > triggers["apparent_temperature_c_lte"]:
                    rule_pass = False
                elif "apparent_temperature_c_lte" in triggers:
                    reasons.append(f"Apparent temp {apparent_temp}°C <= {triggers['apparent_temperature_c_lte']}°C")

                if "temperature_c_gte" in triggers and temp < triggers["temperature_c_gte"]:
                    rule_pass = False
                elif "temperature_c_gte" in triggers:
                    reasons.append(f"Temperature {temp}°C >= {triggers['temperature_c_gte']}°C")

                if "uv_index_gte" in triggers and uv_index < triggers["uv_index_gte"]:
                    rule_pass = False
                elif "uv_index_gte" in triggers:
                    reasons.append(f"UV index {uv_index} >= {triggers['uv_index_gte']}")

                if "precipitation_probability_gte" in triggers and precip_prob < triggers["precipitation_probability_gte"]:
                    rule_pass = False
                elif "precipitation_probability_gte" in triggers:
                    reasons.append(f"Precip probability {precip_prob}% >= {triggers['precipitation_probability_gte']}%")

                if "precipitation_mm_gte" in triggers and precip < triggers["precipitation_mm_gte"]:
                    rule_pass = False
                elif "precipitation_mm_gte" in triggers:
                    reasons.append(f"Precipitation {precip}mm >= {triggers['precipitation_mm_gte']}mm")

                if "relative_humidity_gte" in triggers and humidity < triggers["relative_humidity_gte"]:
                    rule_pass = False
                elif "relative_humidity_gte" in triggers:
                    reasons.append(f"Humidity {humidity}% >= {triggers['relative_humidity_gte']}%")

                condition_met = rule_pass

            # 4. Fuzzy evaluations (e.g. Picnic comfort SOP-011 or Muggy SOP-012)
            elif sop.get("condition_type") == "fuzzy":
                if sop.get("id") == "SOP-011":
                    # Picnic ideal envelope:
                    in_temp = (triggers.get("temperature_c_min", 18.0) <= temp <= triggers.get("temperature_c_max", 28.5))
                    in_precip = (precip <= triggers.get("precipitation_mm_lte", 0.2) and precip_prob <= triggers.get("precipitation_probability_lte", 25))
                    in_wind = (wind_speed <= triggers.get("wind_speed_kmh_lte", 22.0))
                    in_uv = (uv_index <= triggers.get("uv_index_lte", 7.0))
                    if in_temp and in_precip and in_wind and in_uv:
                        condition_met = True
                        reasons.append(f"Fuzzy picnic comfort met: Temp {temp}°C (optimal 18-28.5°C), zero rain, gentle wind {wind_speed}km/h, safe UV {uv_index}.")
                elif sop.get("id") == "SOP-012":
                    # Muggy discomfort
                    is_humid = humidity >= triggers.get("relative_humidity_gte", 85)
                    is_warm = temp >= triggers.get("temperature_c_gte", 29.0)
                    is_dryish = precip <= triggers.get("precipitation_mm_lte", 2.0)
                    if is_humid and is_warm and is_dryish:
                        condition_met = True
                        reasons.append(f"Fuzzy muggy discomfort met: High humidity ({humidity}%) with warm temp ({temp}°C).")

            if condition_met and reasons:
                matched_sops.append({
                    "id": sop["id"],
                    "title": sop["title"],
                    "category": sop["category"],
                    "severity": sop["severity"],
                    "severity_rank": sop.get("severity_rank", 1),
                    "guidance": sop["guidance"],
                    "recommended_actions": sop.get("recommended_actions", []),
                    "rationale": sop.get("rationale", ""),
                    "trigger_reasons": reasons,
                })

        # Conflict Resolution: Rank by severity_rank descending
        matched_sops.sort(key=lambda x: x["severity_rank"], reverse=True)

        if not matched_sops:
            return {
                "matched": False,
                "primary_sop": None,
                "contributing_sops": [],
                "all_matches": []
            }

        return {
            "matched": True,
            "primary_sop": matched_sops[0],
            "contributing_sops": matched_sops[1:3],
            "all_matches": matched_sops
        }
