import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple

SOPS_FILE_PATH = os.path.join(os.path.dirname(__file__), "sops.json")

class SOPEngine:
    """
    Decoupled SOP Policy Evaluation Engine.
    Reads policy definitions from sops.json and evaluates them dynamically against
    live meteorological metrics and user intent.
    
    Zero code modification is required to add or update SOP rules.
    Condition evaluations are 100% data-driven based on the JSON schemas.
    """
    METRIC_ALIASES = {
        "temperature": "temperature_2m",
        "temperature_c": "temperature_2m",
        "temperature_2m": "temperature_2m",
        "temp": "temperature_2m",
        "temp_c": "temperature_2m",
        "apparent_temperature": "apparent_temperature",
        "apparent_temperature_c": "apparent_temperature",
        "apparent_temp": "apparent_temperature",
        "relative_humidity": "relative_humidity_2m",
        "relative_humidity_pct": "relative_humidity_2m",
        "relative_humidity_2m": "relative_humidity_2m",
        "humidity": "relative_humidity_2m",
        "precipitation": "precipitation",
        "precipitation_mm": "precipitation",
        "rain": "rain",
        "rain_mm": "rain",
        "precipitation_probability": "precipitation_probability",
        "precipitation_probability_pct": "precipitation_probability",
        "precip_prob": "precipitation_probability",
        "wind_speed": "wind_speed_10m",
        "wind_speed_kmh": "wind_speed_10m",
        "wind_speed_10m": "wind_speed_10m",
        "wind_gusts": "wind_gusts_10m",
        "wind_gusts_kmh": "wind_gusts_10m",
        "wind_gusts_10m": "wind_gusts_10m",
        "uv_index": "uv_index",
        "uv": "uv_index",
        "weather_code": "weather_code",
    }

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

    def _resolve_metric_value(self, metric_raw: str, weather: Dict[str, Any]) -> Tuple[Optional[float], str]:
        """Maps user or schema metric names to exact telemetry values."""
        clean_key = metric_raw.lower().strip()
        standard_key = self.METRIC_ALIASES.get(clean_key, clean_key)
        val = weather.get(standard_key)
        if val is None:
            # Fallback attempts for apparent temp or rain
            if "apparent" in clean_key:
                val = weather.get("temperature_2m")
            elif "rain" in clean_key or "precip" in clean_key:
                val = weather.get("precipitation", 0.0)
            elif "gust" in clean_key:
                val = weather.get("wind_speed_10m", 0.0)
        
        try:
            return float(val) if val is not None else None, standard_key
        except (ValueError, TypeError):
            return None, standard_key

    def _matches_activity(self, sop: Dict[str, Any], activity: str, query_text: str) -> Tuple[bool, int]:
        """
        Determines if an SOP applies to the requested activity or query.
        Returns (is_match, specificity_score).
        """
        applicable = [t.lower().strip() for t in sop.get("applicable_activities", [])]
        if "all" in applicable:
            return True, 0

        excluded = [e.lower().strip() for e in sop.get("excluded_activities", [])]
        norm_act = (activity or "general").lower().strip()
        norm_query = (query_text or "").lower().strip()

        # Reject if any excluded activity keywords are present
        for exc in excluded:
            if re.search(rf'\b{re.escape(exc)}\b', norm_act) or re.search(rf'\b{re.escape(exc)}\b', norm_query):
                return False, 0

        match_count = 0
        for tag in applicable:
            # Word boundary matching prevents partial substring false matches
            pattern = rf'\b{re.escape(tag)}\b'
            if re.search(pattern, norm_act) or re.search(pattern, norm_query):
                match_count += 1

        return (match_count > 0), match_count

    def _evaluate_single_trigger(
        self,
        trigger_key: str,
        threshold: Any,
        weather: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Evaluates a single metric comparator rule against weather data."""
        clean_key = trigger_key.lower().strip()

        # Set membership (weather codes)
        if clean_key in ["weather_codes", "weather_code_in", "codes"]:
            current_code = int(weather.get("weather_code", 0))
            if isinstance(threshold, list) and current_code in threshold:
                return True, f"Weather code {current_code} in convective/storm codes {threshold}"
            return False, None

        # Operator extraction
        op = None
        metric_name = clean_key
        for suffix, operator in [
            ("_gte", ">="),
            ("_lte", "<="),
            ("_gt", ">"),
            ("_lt", "<"),
            ("_eq", "=="),
            ("_min", ">="),
            ("_max", "<="),
        ]:
            if clean_key.endswith(suffix):
                op = operator
                metric_name = clean_key[:-len(suffix)]
                break

        if not op:
            return False, None

        current_val, std_name = self._resolve_metric_value(metric_name, weather)
        if current_val is None:
            return False, None

        target_thresh = float(threshold)
        passed = False
        if op == ">=":
            passed = (current_val >= target_thresh)
        elif op == "<=":
            passed = (current_val <= target_thresh)
        elif op == ">":
            passed = (current_val > target_thresh)
        elif op == "<":
            passed = (current_val < target_thresh)
        elif op == "==":
            passed = (current_val == target_thresh)

        if passed:
            reason = f"{std_name} ({current_val}) {op} threshold ({target_thresh})"
            return True, reason
        return False, None

    def _evaluate_clause(
        self,
        clause: Dict[str, Any],
        weather: Dict[str, Any],
        query_text: str
    ) -> Tuple[bool, List[str]]:
        """Evaluates an individual condition clause containing triggers and optional system flags."""
        triggers = clause.get("triggers", {})
        system_flags = clause.get("system_flags", [])
        logic = clause.get("condition_logic", "all").lower()

        reasons = []
        clause_results = []

        # System / context keywords
        if system_flags:
            norm_q = (query_text or "").lower()
            flags_matched = [f for f in system_flags if f.lower() in norm_q]
            if flags_matched:
                clause_results.append(True)
                reasons.append(f"Contextual system alert flags active: {', '.join(flags_matched)}")
            else:
                clause_results.append(False)

        # Trigger metrics
        for t_key, t_thresh in triggers.items():
            pass_single, reason_str = self._evaluate_single_trigger(t_key, t_thresh, weather)
            clause_results.append(pass_single)
            if pass_single and reason_str:
                reasons.append(reason_str)

        if not clause_results:
            return False, []

        if logic == "any":
            clause_passed = any(clause_results)
        else:
            clause_passed = all(clause_results)

        return clause_passed, (reasons if clause_passed else [])

    def evaluate(
        self,
        weather: Dict[str, Any],
        activity: str,
        query_text: str = ""
    ) -> Dict[str, Any]:
        """
        Evaluates active SOPs dynamically against current live weather metrics and user activity.
        100% data-driven condition logic without hardcoded SOP identifiers.
        """
        sops = self._load_sops()
        matched_sops = []

        for sop in sops:
            applies, spec_score = self._matches_activity(sop, activity, query_text)
            if not applies:
                continue

            sop_passed = False
            sop_reasons = []

            # 1. Multi-clause condition logic
            clauses = sop.get("clauses")
            if clauses and isinstance(clauses, list):
                clause_op = sop.get("clause_operator", "any").lower()
                evaluated_clauses = [
                    self._evaluate_clause(c, weather, query_text)
                    for c in clauses
                ]
                clause_statuses = [res[0] for res in evaluated_clauses]
                
                if clause_op == "all":
                    sop_passed = all(clause_statuses)
                else:  # "any"
                    sop_passed = any(clause_statuses)

                if sop_passed:
                    for passed, c_reasons in evaluated_clauses:
                        if passed:
                            sop_reasons.extend(c_reasons)

            # 2. Direct top-level triggers
            elif "triggers" in sop:
                direct_clause = {
                    "triggers": sop.get("triggers", {}),
                    "system_flags": sop.get("system_flags", []),
                    "condition_logic": sop.get("condition_logic", "all")
                }
                sop_passed, sop_reasons = self._evaluate_clause(direct_clause, weather, query_text)

            if sop_passed and sop_reasons:
                matched_sops.append({
                    "id": sop["id"],
                    "title": sop["title"],
                    "category": sop["category"],
                    "severity": sop["severity"],
                    "severity_rank": sop.get("severity_rank", 1),
                    "guidance": sop["guidance"],
                    "recommended_actions": sop.get("recommended_actions", []),
                    "rationale": sop.get("rationale", ""),
                    "trigger_reasons": sop_reasons,
                    "specificity_score": spec_score,
                })

        # Conflict Resolution:
        # Sort primarily by severity_rank descending (CRITICAL -> WARNING -> CAUTION -> ADVISORY),
        # secondarily by activity specificity score.
        matched_sops.sort(
            key=lambda x: (x["severity_rank"], x["specificity_score"]),
            reverse=True
        )

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
