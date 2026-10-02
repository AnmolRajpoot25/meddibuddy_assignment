"""Quick validation that TelemetryValidator behaves correctly before running full eval."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.telemetry_validator import TelemetryValidator, VALID_SOP_IDS

weather = {
    "temperature_2m": 41.0, "apparent_temperature": 47.0,
    "wind_speed_10m": 14.0, "wind_gusts_10m": 20.0,
    "precipitation": 0.0
}
sop = {
    "id": "SOP-004",
    "guidance": "Heat index 38 degrees C or higher.",
    "triggers": {"apparent_temperature_c_gte": 38.0},
    "clauses": []
}

print("Valid SOP IDs:", sorted(VALID_SOP_IDS))
print()

# Test 1: Grounded response with actual telemetry values -> should PASS
resp_grounded = (
    "[SOP-004] Severity: WARNING\n"
    "Temperature: 41.0 degrees C (Apparent: 47.0 degrees C)\n"
    "Wind Speed: 14.0 km/h (Gusts: 20.0 km/h)\n"
    "Precipitation: 0.0 mm"
)
ok, viols = TelemetryValidator.validate(resp_grounded, weather, sop)
print(f"[{'PASS' if ok else 'FAIL'}] Test 1 - Grounded response (should pass): violations={viols}")

# Test 2: Policy threshold value (38C) cited in guidance context -> should PASS
resp_threshold = "[SOP-004] The heat index threshold is 38 degrees C. Current temperature: 41.0 degrees C"
ok2, viols2 = TelemetryValidator.validate(resp_threshold, weather, sop)
print(f"[{'PASS' if ok2 else 'FAIL'}] Test 2 - Policy threshold 38C cited (should pass): violations={viols2}")

# Test 3: Fabricated temperature (25C vs actual 41C) -> should FAIL
resp_hallucinated = "[SOP-004] Temperature: 25.0 degrees C (Apparent: 27.0 degrees C)"
ok3, viols3 = TelemetryValidator.validate(resp_hallucinated, weather, sop)
print(f"[{'PASS' if not ok3 else 'FAIL'}] Test 3 - Hallucinated 25C (should be caught): violations={viols3}")

# Test 4: Unauthorized SOP-999 -> should FAIL
resp_bad = "[SOP-004] advisory. Also see SOP-999 for more details."
ok4, viols4 = TelemetryValidator.validate(resp_bad, weather, sop)
print(f"[{'PASS' if not ok4 else 'FAIL'}] Test 4 - SOP-999 hallucination (should be caught): violations={viols4}")

# Test 5: SOP-003 cited alongside allowed primary SOP -> should PASS (SOP-003 is in sops.json)
resp_valid_secondary = "[SOP-004] advisory. Contributing: [SOP-005] also applies."
sop_contrib = [{"id": "SOP-005", "triggers": {}, "clauses": []}]
ok5, viols5 = TelemetryValidator.validate(resp_valid_secondary, weather, sop, contributing_sops=sop_contrib)
print(f"[{'PASS' if ok5 else 'FAIL'}] Test 5 - SOP-005 as contributing (should pass): violations={viols5}")

print()
print("All validator unit tests complete.")
