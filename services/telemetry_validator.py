import re
import json
import os
import logging
from typing import Dict, Any, List, Tuple, Optional, Set

logger = logging.getLogger(__name__)

# Pre-load the set of valid SOP IDs from sops.json so we can validate citations
_SOPS_FILE = os.path.join(os.path.dirname(__file__), "..", "policies", "sops.json")

def _load_valid_sop_ids() -> Set[str]:
    try:
        with open(_SOPS_FILE, "r", encoding="utf-8") as f:
            sops = json.load(f)
        return {s["id"].upper() for s in sops if "id" in s}
    except Exception:
        return set()

VALID_SOP_IDS: Set[str] = _load_valid_sop_ids()


class TelemetryValidator:
    """
    Zero-Hallucination Post-LLM Validator.

    Checks three things on every LLM response:
    1. The mandated primary SOP ID is cited.
    2. No unauthorized / hallucinated SOP IDs appear (anything not in sops.json).
    3. Every weather metric cited in an *observed / current reading* context matches
       the verified Open-Meteo telemetry (tolerance: ±1 °C temp, ±2 km/h wind,
       ±1 mm precip). Policy-guidance threshold values (e.g. "38°C threshold")
       are explicitly allowed and NOT treated as telemetry claims.
    """

    # Patterns that signal an *observed / current* weather figure
    # (i.e. the LLM is reporting what the sensor measured, not quoting a policy limit)
    # Each pattern captures the numeric value in group 1.
    # Note: case-insensitive flag applied at match time.
    _CURRENT_TEMP_PATTERNS = [
        # "current/observed/actual temperature is X°C" or "...X degrees C"
        r'(?:current|observed|actual|recorded|live|measured)[^\n]{0,60}?(-?\d+(?:\.\d+)?)\s*(?:°\s*[cC]|degrees?\s+[cC](?:\b|elsius))',
        # Markdown bold label: **Temperature:** or **Temp:** followed by value
        r'\*\*[Tt]emp(?:erature)?(?:\*\*|:)[^\n]{0,30}?(-?\d+(?:\.\d+)?)\s*(?:°\s*[cC]|degrees?\s+[cC](?:\b|elsius))',
        # Label then colon then value on same line: "Temperature: 41.0°C"
        r'[Tt]emp(?:erature)?\s*(?:[^:\n]{0,5})?:\s*(-?\d+(?:\.\d+)?)\s*(?:°\s*[cC]|degrees?\s+[cC](?:\b|elsius))',
        # "temp ... is X°C"
        r'[Tt]emp(?:erature)?[^\n]{0,15}?\bis\s+(-?\d+(?:\.\d+)?)\s*(?:°\s*[cC]|degrees?\s+[cC](?:\b|elsius))',
    ]

    _CURRENT_WIND_PATTERNS = [
        # "current/observed wind speed is X km/h"
        r'(?:current|observed|actual|recorded|live|measured)[^\n]{0,60}?(\d+(?:\.\d+)?)\s*km/h',
        # Markdown bold label: **Wind Speed:** X km/h  (must be > 0 to skip Gusts: 0.0 false-positives)
        r'\*\*[Ww]ind\s*[Ss]peed\*\*[^\n]{0,30}?(\d+(?:\.\d+)?)\s*km/h',
        # Label then colon: "Wind Speed: 14.0 km/h" or "Wind speed: ..."
        r'[Ww]ind\s+[Ss]peed\s*(?:[^:\n]{0,5})?:\s*(\d+(?:\.\d+)?)\s*km/h',
        # "wind speed is X km/h"
        r'[Ww]ind\s+[Ss]peed[^\n]{0,15}?\bis\s+(\d+(?:\.\d+)?)\s*km/h',
    ]

    _CURRENT_PRECIP_PATTERNS = [
        # "current/observed precipitation is X mm"
        r'(?:current|observed|actual|recorded|live|measured)[^\n]{0,60}?(\d+(?:\.\d+)?)\s*mm(?:\b)',
        # Markdown bold: **Precipitation:** X mm
        r'\*\*[Pp]recipitation\*\*[^\n]{0,30}?(\d+(?:\.\d+)?)\s*mm(?:\b)',
        # Label then colon: "Precipitation: 0.0 mm"
        r'[Pp]recipitation\s*(?:[^:\n]{0,5})?:\s*(\d+(?:\.\d+)?)\s*mm(?:\b)',
        # "precipitation is X mm"
        r'[Pp]recipitation[^\n]{0,15}?\bis\s+(\d+(?:\.\d+)?)\s*mm(?:\b)',
    ]

    @staticmethod
    def _collect_policy_values(primary_sop: Optional[Dict], contributing_sops: Optional[List[Dict]]) -> Set[float]:
        """Gather all numeric threshold values declared in policy triggers so we can exempt them."""
        policy_vals: Set[float] = set()
        sops_to_check = []
        if primary_sop:
            sops_to_check.append(primary_sop)
        if contributing_sops:
            sops_to_check.extend(contributing_sops)
        for sop in sops_to_check:
            triggers = sop.get("triggers", {})
            for v in triggers.values():
                if isinstance(v, (int, float)):
                    policy_vals.add(float(v))
            for clause in sop.get("clauses", []):
                for v in clause.get("triggers", {}).values():
                    if isinstance(v, (int, float)):
                        policy_vals.add(float(v))
        return policy_vals

    @staticmethod
    def _extract_with_patterns(patterns: List[str], text: str) -> List[float]:
        values = []
        for pat in patterns:
            for m in re.finditer(pat, text, re.IGNORECASE):
                try:
                    values.append(float(m.group(1)))
                except (ValueError, IndexError):
                    pass
        return values

    @staticmethod
    def _matches_formatting_tolerance(cited: float, verified: float) -> bool:
        """
        Formatting-level tolerance only:
        - Exact float match (abs diff < 0.05)
        - Integer rounding match if cited without decimal (e.g. 23 vs 23.0 or 23 vs 22.6, abs diff <= 0.5)
        Eliminates wide numerical drift margins.
        """
        if abs(cited - verified) < 0.05:
            return True
        if cited == round(cited) and abs(cited - verified) <= 0.5:
            return True
        return False

    @staticmethod
    def format_verified_telemetry_block(location_name: str, weather: Dict[str, Any]) -> str:
        """Application-generated verified telemetry block. Preferred over LLM generation."""
        temp = weather.get("temperature_2m", "N/A")
        app_temp = weather.get("apparent_temperature", temp)
        wind = weather.get("wind_speed_10m", "N/A")
        gusts = weather.get("wind_gusts_10m", wind)
        precip = weather.get("precipitation", "N/A")
        precip_prob = weather.get("precipitation_probability", "N/A")
        rh = weather.get("relative_humidity_2m", "N/A")
        uv = weather.get("uv_index", "N/A")
        obs_time = weather.get("time", "Current")

        return (
            f"**Verified Meteorological Observations ({location_name}, {obs_time}):**\n"
            f"- Temperature: {temp}°C (Apparent: {app_temp}°C)\n"
            f"- Precipitation: {precip} mm (Probability: {precip_prob}%)\n"
            f"- Wind Speed: {wind} km/h (Gusts: {gusts} km/h)\n"
            f"- Relative Humidity: {rh}%\n"
            f"- UV Index: {uv}"
        )

    @staticmethod
    def validate(
        response_text: str,
        weather: Dict[str, Any],
        primary_sop: Dict[str, Any],
        contributing_sops: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[bool, List[str]]:
        violations: List[str] = []

        # ── 1. SOP traceability ────────────────────────────────────────────────
        sop_id = (primary_sop or {}).get("id", "")
        if sop_id and sop_id not in response_text:
            violations.append(f"Missing mandatory primary SOP ID citation: [{sop_id}]")

        # ── 2. Hallucinated / unauthorized SOP IDs ────────────────────────────
        allowed_ids: Set[str] = set()
        if sop_id:
            allowed_ids.add(sop_id.upper())
        if contributing_sops:
            for cs in contributing_sops:
                if cs.get("id"):
                    allowed_ids.add(cs["id"].upper())

        # Reload valid IDs in case sops.json was modified at runtime
        valid_ids = _load_valid_sop_ids() or VALID_SOP_IDS

        cited_ids = set(re.findall(r'\bSOP-\d{3,}\b', response_text, re.I))
        for cited in cited_ids:
            cited_upper = cited.upper()
            if cited_upper not in allowed_ids and cited_upper not in valid_ids:
                violations.append(
                    f"Response cited unauthorized/hallucinated policy: [{cited_upper}] "
                    f"(not in sops.json and not in approved response set)"
                )

        # ── 3. Telemetry number grounding (formatting tolerance only) ─────────
        policy_values = TelemetryValidator._collect_policy_values(primary_sop, contributing_sops)

        def _is_policy_value(val: float) -> bool:
            return any(abs(val - pv) < 0.05 or (val == round(val) and abs(val - pv) <= 0.5) for pv in policy_values)

        # 3a. Temperature — only check values cited in *current reading* context
        actual_temp = weather.get("temperature_2m")
        actual_app = weather.get("apparent_temperature", actual_temp)
        valid_temps = [float(v) for v in [actual_temp, actual_app] if v is not None]

        cited_temps = TelemetryValidator._extract_with_patterns(
            TelemetryValidator._CURRENT_TEMP_PATTERNS,
            response_text
        )
        for t in cited_temps:
            if _is_policy_value(t):
                continue
            if valid_temps and not any(TelemetryValidator._matches_formatting_tolerance(t, vt) for vt in valid_temps):
                violations.append(
                    f"Telemetry mismatch — temperature cited: {t}°C "
                    f"(verified: temp={actual_temp}°C, apparent={actual_app}°C, formatting tolerance only)"
                )

        # 3b. Wind speed — only check values in *current reading* context
        actual_wind = weather.get("wind_speed_10m")
        actual_gusts = weather.get("wind_gusts_10m", actual_wind)
        valid_winds = [float(v) for v in [actual_wind, actual_gusts] if v is not None]

        cited_winds = TelemetryValidator._extract_with_patterns(
            TelemetryValidator._CURRENT_WIND_PATTERNS, response_text
        )
        for w in cited_winds:
            if _is_policy_value(w):
                continue
            if valid_winds and not any(TelemetryValidator._matches_formatting_tolerance(w, vw) for vw in valid_winds):
                violations.append(
                    f"Telemetry mismatch — wind cited: {w} km/h "
                    f"(verified: {actual_wind} km/h sustained, {actual_gusts} km/h gusts, formatting tolerance only)"
                )

        # 3c. Precipitation — only check values in *current reading* context
        actual_precip = weather.get("precipitation")
        cited_precips = TelemetryValidator._extract_with_patterns(
            TelemetryValidator._CURRENT_PRECIP_PATTERNS, response_text
        )
        if actual_precip is not None:
            ap = float(actual_precip)
            for p in cited_precips:
                if _is_policy_value(p):
                    continue
                if not TelemetryValidator._matches_formatting_tolerance(p, ap):
                    violations.append(
                        f"Telemetry mismatch — precipitation cited: {p} mm "
                        f"(verified: {ap} mm, formatting tolerance only)"
                    )

        return (len(violations) == 0), violations
