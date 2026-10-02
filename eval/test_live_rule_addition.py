import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from policies.sop_engine import SOPEngine, SOPS_FILE_PATH

def test_live_sop_addition():
    print("=" * 70)
    print("  VERIFYING DYNAMIC SOP ADDITION (ZERO CODE CHANGES TEST)")
    print("=" * 70)

    engine = SOPEngine()
    initial_sops = engine.get_all_sops()
    initial_count = len(initial_sops)
    print(f"1. Baseline: Loaded {initial_count} SOPs from {SOPS_FILE_PATH}")

    # Read file
    with open(SOPS_FILE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 13th SOP simulation: e.g. Kite Flying Hazard under High Wind
    new_sop = {
        "id": "SOP-013-DEMO",
        "title": "Kite Flying Hazard under Gusty Wind",
        "category": "leisure_and_events",
        "severity": "WARNING",
        "severity_rank": 3,
        "applicable_activities": ["kite flying", "kite", "kiting"],
        "condition_type": "numeric",
        "triggers": {
            "wind_speed_kmh_gte": 25.0
        },
        "guidance": "High winds exceeding 25 km/h make kite control erratic, posing cutting injuries from glass-coated strings (manja) and entanglement in electrical cables.",
        "recommended_actions": [
            "Do not fly kites near electrical substations or high-tension wires.",
            "Postpone kite flying until wind speeds drop below 20 km/h."
        ],
        "rationale": "High tension cord breakage and electrical arcing hazards."
    }

    try:
        # Append without modifying ANY Python code
        data.append(new_sop)
        # Small sleep so mtime updates
        time.sleep(1)
        with open(SOPS_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Check engine without restarting or re-instantiating with code
        updated_sops = engine.get_all_sops()
        updated_count = len(updated_sops)
        print(f"2. Dynamically reloaded: Found {updated_count} SOPs (added {new_sop['id']})")

        # Test evaluation against simulated weather
        test_weather = {"wind_speed_10m": 28.0, "temperature_2m": 24.0}
        eval_res = engine.evaluate(test_weather, activity="kite flying", query_text="Is it safe to fly a kite today?")

        print(f"3. Evaluation of new SOP without code edit: Matched = {eval_res.get('matched')}")
        if eval_res.get("matched"):
            print(f"   Primary SOP Matched: [{eval_res['primary_sop']['id']}] {eval_res['primary_sop']['title']}")
            print(f"   Severity: {eval_res['primary_sop']['severity']}")
            print(f"   Guidance: {eval_res['primary_sop']['guidance']}")
            print("\nRESULT: SUCCESS! New policy was applied with ZERO code changes.")
        else:
            print("\nRESULT: FAILED to match new SOP.")

    finally:
        # Revert sops.json to clean state
        clean_data = [s for s in data if s["id"] != "SOP-013-DEMO"]
        with open(SOPS_FILE_PATH, "w", encoding="utf-8") as f:
            json.dump(clean_data, f, indent=2)
        print(f"4. Reverted {SOPS_FILE_PATH} back to original {initial_count} SOPs.")

if __name__ == "__main__":
    test_live_sop_addition()
