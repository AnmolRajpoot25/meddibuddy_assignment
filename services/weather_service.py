import requests
from typing import Dict, Any, Optional
from datetime import datetime

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

class WeatherService:
    """
    Client for Open-Meteo Geocoding and Weather Forecast APIs.
    Retrieves verified live meteorological observations and forecasts.
    Enforces honest error handling when coordinates cannot be resolved
    or when endpoints are unreachable.
    """
    def __init__(self, timeout: int = 8):
        self.timeout = timeout

    def resolve_location(self, city_name: str) -> Dict[str, Any]:
        """
        Resolves city name to latitude and longitude via Open-Meteo Geocoding API.
        Returns candidate details or raises/returns failure.
        """
        if not city_name or not city_name.strip():
            return {"success": False, "error": "No city name provided"}

        cleaned_city = city_name.strip()
        params = {
            "name": cleaned_city,
            "count": 5,
            "language": "en",
            "format": "json"
        }

        try:
            resp = requests.get(GEOCODING_URL, params=params, timeout=self.timeout)
            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"Geocoding service returned HTTP status {resp.status_code}"
                }
            data = resp.json()
            results = data.get("results")
            if not results:
                return {
                    "success": False,
                    "error": f"Could not resolve any geographical location for '{city_name}'"
                }

            # Select primary match
            first_match = results[0]
            return {
                "success": True,
                "city": first_match.get("name"),
                "latitude": first_match.get("latitude"),
                "longitude": first_match.get("longitude"),
                "country": first_match.get("country", ""),
                "admin1": first_match.get("admin1", ""),
                "timezone": first_match.get("timezone", "auto")
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Geocoding network error: {str(e)}"
            }

    def fetch_weather(
        self,
        latitude: float,
        longitude: float,
        timeframe: Optional[str] = None,
        simulate_failure: bool = False
    ) -> Dict[str, Any]:
        """
        Pulls live weather metrics from Open-Meteo Forecast endpoint.
        Explicitly requests current, hourly fields, and handles timeframe offsets.
        """
        if simulate_failure:
            return {
                "success": False,
                "error": "Simulated Open-Meteo endpoint unreachable (503 Service Unavailable)"
            }

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,rain,weather_code,wind_speed_10m,wind_gusts_10m,uv_index",
            "hourly": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation_probability,precipitation,wind_speed_10m,uv_index",
            "timezone": "auto"
        }

        try:
            resp = requests.get(FORECAST_URL, params=params, timeout=self.timeout)
            if resp.status_code != 200:
                return {
                    "success": False,
                    "error": f"Open-Meteo weather endpoint returned status {resp.status_code}"
                }

            data = resp.json()
            current = data.get("current", {})
            hourly = data.get("hourly", {})

            if not current:
                return {
                    "success": False,
                    "error": "Open-Meteo returned empty current weather payload"
                }

            # Baseline metrics from current
            temp = float(current.get("temperature_2m", 0.0))
            apparent_temp = float(current.get("apparent_temperature", temp))
            humidity = float(current.get("relative_humidity_2m", 50.0))
            precip = float(current.get("precipitation", 0.0))
            rain = float(current.get("rain", 0.0))
            wind_speed = float(current.get("wind_speed_10m", 0.0))
            wind_gusts = float(current.get("wind_gusts_10m", wind_speed))
            uv_index = float(current.get("uv_index", 0.0))
            precip_prob = float(current.get("precipitation_probability", 0.0))
            obs_time = current.get("time")

            # If user queried a specific timeframe (e.g. afternoon / 1:00 PM / evening),
            # pull corresponding hourly forecast values
            if timeframe and hourly:
                target_hour = None
                tf_lower = timeframe.lower()
                if "1:00 pm" in tf_lower or "13:00" in tf_lower or "afternoon" in tf_lower or "midday" in tf_lower:
                    target_hour = 13
                elif "evening" in tf_lower or "tonight" in tf_lower:
                    target_hour = 19
                elif "morning" in tf_lower:
                    target_hour = 9

                if target_hour is not None and hourly.get("time") and len(hourly["time"]) > target_hour:
                    if hourly.get("temperature_2m"):
                        temp = float(hourly["temperature_2m"][target_hour])
                    if hourly.get("apparent_temperature"):
                        apparent_temp = float(hourly["apparent_temperature"][target_hour])
                    if hourly.get("relative_humidity_2m"):
                        humidity = float(hourly["relative_humidity_2m"][target_hour])
                    if hourly.get("precipitation_probability"):
                        precip_prob = float(hourly["precipitation_probability"][target_hour])
                    if hourly.get("precipitation"):
                        precip = float(hourly["precipitation"][target_hour])
                    if hourly.get("wind_speed_10m"):
                        wind_speed = float(hourly["wind_speed_10m"][target_hour])
                    if hourly.get("uv_index"):
                        uv_index = float(hourly["uv_index"][target_hour])
                    obs_time = hourly["time"][target_hour]
            elif not precip_prob and hourly.get("precipitation_probability"):
                precip_prob = float(hourly["precipitation_probability"][0])

            return {
                "success": True,
                "time": obs_time,
                "temperature_2m": temp,
                "apparent_temperature": apparent_temp,
                "relative_humidity_2m": humidity,
                "precipitation": precip,
                "rain": rain,
                "precipitation_probability": precip_prob,
                "wind_speed_10m": wind_speed,
                "wind_gusts_10m": wind_gusts,
                "uv_index": uv_index,
                "weather_code": int(current.get("weather_code", 0)),
                "raw_current": current
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Failed to connect to Open-Meteo weather service: {str(e)}"
            }

    def get_weather_for_city(
        self,
        city_name: str,
        timeframe: Optional[str] = None,
        simulate_failure: bool = False
    ) -> Dict[str, Any]:
        """Convenience method resolving location and fetching weather in one flow."""
        loc_res = self.resolve_location(city_name)
        if not loc_res.get("success"):
            return {
                "success": False,
                "stage": "geocoding",
                "error": loc_res.get("error")
            }

        weather_res = self.fetch_weather(
            latitude=loc_res["latitude"],
            longitude=loc_res["longitude"],
            timeframe=timeframe,
            simulate_failure=simulate_failure
        )
        if not weather_res.get("success"):
            return {
                "success": False,
                "stage": "forecast",
                "location": loc_res,
                "error": weather_res.get("error")
            }

        return {
            "success": True,
            "location": loc_res,
            "weather": weather_res
        }
