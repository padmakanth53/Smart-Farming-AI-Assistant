#!/usr/bin/env python3
"""
Advanced Weather Module for Smart Farming AI Assistant
Provides weather data, alerts, and farming recommendations
Compatible with Python 3.10+ and PyCharm 2023
"""

import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import random
import time

logger = logging.getLogger(__name__)

class WeatherService:
    """Advanced weather service for farming applications"""

    def __init__(self, api_key: Optional[str] = "51c36a39b3b78e626ab0d4daa29c12f4"):
        """
        Initialize weather service

        Args:
            api_key: OpenWeatherMap API key (optional for demo mode)
        """
        self.api_key = api_key
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.use_demo_mode = not api_key

        if self.use_demo_mode:
            logger.info("Weather service running in demo mode")
        else:
            logger.info("Weather service initialized with API key")

        # Define default weather thresholds for farming
        self.farming_thresholds = {
            'temp_high_warning': 35,    # Celsius
            'temp_critical_alert': 40,  # Celsius
            'temp_low_warning': 10,     # Celsius
            'temp_frost_alert': 5,      # Celsius
            'humidity_high_warning': 85, # Percentage
            'humidity_low_warning': 40,  # Percentage
            'wind_speed_high_warning': 20, # km/h
            'wind_speed_critical_alert': 30, # km/h
            'rainfall_heavy_warning': 25, # mm in 24 hours
            'rainfall_critical_alert': 50 # mm in 24 hours
        }


    def _get_api_weather(self, location: str) -> Dict:
        """Fetch weather data from OpenWeatherMap API."""
        if not self.api_key:
            raise ValueError("API Key is required for non-demo mode.")

        # Geocoding to get coordinates
        geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={location}&limit=1&appid={self.api_key}"
        try:
            geo_response = requests.get(geo_url)
            geo_response.raise_for_status()
            geo_data = geo_response.json()
            if not geo_data:
                raise ValueError(f"Could not find coordinates for location: {location}")
            lat = geo_data[0]['lat']
            lon = geo_data[0]['lon']
        except requests.exceptions.RequestException as e:
            logger.error(f"Geocoding API request failed: {e}")
            return {}
        except ValueError as e:
            logger.error(e)
            return {}

        # Fetch current weather
        weather_url = f"{self.base_url}/weather?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
        try:
            response = requests.get(weather_url)
            response.raise_for_status()
            data = response.json()
            logger.info(f"Successfully fetched weather for {location} from API.")
            return {
                'location': data.get('name'),
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind']['speed'] * 3.6, # Convert m/s to km/h
                'conditions': data['weather'][0]['description'],
                'icon': data['weather'][0]['icon'],
                'timestamp': datetime.fromtimestamp(data['dt']).isoformat(),
                'sunrise': datetime.fromtimestamp(data['sys']['sunrise']).isoformat(),
                'sunset': datetime.fromtimestamp(data['sys']['sunset']).isoformat(),
                'visibility': data.get('visibility'),
                'cloudiness': data['clouds']['all'],
                'timezone_offset': data['timezone']
            }
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenWeatherMap API request failed: {e}")
            return {}

    def _get_demo_weather(self, location: str) -> Dict:
        """Generate random weather data for demo mode."""
        logger.info(f"Generating demo weather for {location}")
        temp = round(random.uniform(15, 35), 1)
        humidity = random.randint(40, 90)
        wind_speed = round(random.uniform(5, 25), 1)
        conditions_list = [
            "clear sky", "few clouds", "scattered clouds", "broken clouds",
            "shower rain", "rain", "thunderstorm", "snow", "mist"
        ]
        random_condition = random.choice(conditions_list)
        rainfall = 0
        if "rain" in random_condition or "shower" in random_condition:
            rainfall = round(random.uniform(0, 30), 1) # Simulate some rainfall
        elif "thunderstorm" in random_condition:
            rainfall = round(random.uniform(10, 60), 1) # Heavier rainfall for thunderstorms

        current_time = datetime.now()
        sunrise_time = (current_time.replace(hour=6, minute=0, second=0, microsecond=0) + timedelta(hours=random.randint(-1,1))).isoformat()
        sunset_time = (current_time.replace(hour=18, minute=30, second=0, microsecond=0) + timedelta(hours=random.randint(-1,1))).isoformat()


        return {
            'location': location,
            'temperature': temp,
            'feels_like': round(temp + random.uniform(-3, 3), 1),
            'humidity': humidity,
            'pressure': random.randint(1000, 1020),
            'wind_speed': wind_speed,
            'conditions': random_condition,
            'icon': '01d', # Generic icon for demo
            'timestamp': current_time.isoformat(),
            'sunrise': sunrise_time,
            'sunset': sunset_time,
            'visibility': random.randint(5000, 10000),
            'cloudiness': random.randint(0, 100),
            'rainfall': rainfall, # Added rainfall
            'timezone_offset': 19800 # IST offset
        }

    def get_current_weather(self, location: str = "Vellore, Tamil Nadu, IN") -> Dict:
        """
        Get current weather data, using API or demo mode.
        Also generates alerts based on defined farming thresholds.

        Args:
            location: Location string (city, state, country)

        Returns:
            A dictionary containing weather data and a list of alerts.
        """
        weather_data = {}
        if self.use_demo_mode:
            weather_data = self._get_demo_weather(location)
        else:
            weather_data = self._get_api_weather(location)

        if weather_data:
            weather_data['alerts'] = self._check_weather_alerts(weather_data)
        else:
            weather_data['alerts'] = [{
                'type': 'error',
                'message': 'Could not retrieve weather data.',
                'icon': '⚠️'
            }]

        return weather_data


    def _check_weather_alerts(self, weather_data: Dict) -> List[Dict]:
        """
        Check weather conditions against farming thresholds and generate alerts.
        This method is now part of WeatherService.
        """
        alerts = []
        temp = weather_data.get('temperature')
        humidity = weather_data.get('humidity')
        wind_speed = weather_data.get('wind_speed')
        rainfall = weather_data.get('rainfall', 0) # Rainfall can be 0 or missing from API initially

        # Temperature alerts
        if temp is not None:
            if temp >= self.farming_thresholds['temp_critical_alert']:
                alerts.append({
                    'type': 'critical',
                    'message': f"Critical: Extreme high temperature ({temp}°C)! Protect crops from heat stress.",
                    'icon': '🔥'
                })
            elif temp >= self.farming_thresholds['temp_high_warning']:
                alerts.append({
                    'type': 'warning',
                    'message': f"Warning: High temperature ({temp}°C). Increase irrigation frequency.",
                    'icon': '🌡️'
                })
            elif temp <= self.farming_thresholds['temp_frost_alert']:
                alerts.append({
                    'type': 'critical',
                    'message': f"Critical: Frost warning ({temp}°C)! Take immediate measures to protect sensitive crops.",
                    'icon': '🥶'
                })
            elif temp <= self.farming_thresholds['temp_low_warning']:
                alerts.append({
                    'type': 'warning',
                    'message': f"Warning: Low temperature ({temp}°C). Monitor crops for cold stress.",
                    'icon': '❄️'
                })

        # Humidity alerts
        if humidity is not None:
            if humidity >= self.farming_thresholds['humidity_high_warning']:
                alerts.append({
                    'type': 'warning',
                    'message': f"High humidity ({humidity}%) detected. Increased risk of fungal diseases. Ensure good air circulation.",
                    'icon': '💧'
                })
            elif humidity <= self.farming_thresholds['humidity_low_warning']:
                alerts.append({
                    'type': 'info',
                    'message': f"Low humidity ({humidity}%). Consider light irrigation or misting for sensitive crops.",
                    'icon': ' خشک' # Dry icon
                })

        # Wind speed alerts
        if wind_speed is not None:
            if wind_speed >= self.farming_thresholds['wind_speed_critical_alert']:
                alerts.append({
                    'type': 'critical',
                    'message': f"Critical: High winds ({wind_speed} km/h)! Secure unstable crops and structures.",
                    'icon': '🌪️'
                })
            elif wind_speed >= self.farming_thresholds['wind_speed_high_warning']:
                alerts.append({
                    'type': 'warning',
                    'message': f"Strong winds ({wind_speed} km/h). Check for crop damage and soil erosion.",
                    'icon': '💨'
                })

        # Rainfall alerts (assuming rainfall is for a relevant period, e.g., 24h forecast or current intensity)
        if rainfall is not None:
            if rainfall >= self.farming_thresholds['rainfall_critical_alert']:
                alerts.append({
                    'type': 'critical',
                    'message': f"Critical: Heavy rainfall ({rainfall}mm) expected/occurring! Ensure drainage to prevent waterlogging and nutrient loss.",
                    'icon': ' inundaciones' # Flooding icon
                })
            elif rainfall >= self.farming_thresholds['rainfall_heavy_warning']:
                alerts.append({
                    'type': 'warning',
                    'message': f"Significant rainfall ({rainfall}mm) expected/occurring. Monitor soil moisture and drainage.",
                    'icon': '🌧️'
                })

        if not alerts:
            alerts.append({
                'type': 'success',
                'message': 'Weather conditions are favorable for farming activities.',
                'icon': '✅'
            })

        return alerts

    def get_weather_forecast(self, location: str = "Vellore, Tamil Nadu, IN", days: int = 5) -> Dict:
        """
        Get weather forecast data.

        Args:
            location: Location string
            days: Number of days for forecast (1-5 for demo)

        Returns:
            A dictionary containing forecast data.
        """
        if self.use_demo_mode:
            logger.info(f"Generating demo forecast for {location} for {days} days.")
            forecast_data = []
            for i in range(days):
                date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
                min_temp = round(random.uniform(15, 25), 1)
                max_temp = round(random.uniform(25, 35), 1)
                humidity = random.randint(50, 80)
                wind_speed = round(random.uniform(5, 20), 1)
                conditions_list = ["clear sky", "light clouds", "moderate rain", "partly cloudy"]
                random_condition = random.choice(conditions_list)
                daily_rainfall = round(random.uniform(0, 15), 1) if "rain" in random_condition else 0

                forecast_data.append({
                    'date': date,
                    'min_temp': min_temp,
                    'max_temp': max_temp,
                    'humidity': humidity,
                    'wind_speed': wind_speed,
                    'conditions': random_condition,
                    'rainfall': daily_rainfall,
                    'alerts': [] # Alerts could be generated per day for forecast too if needed
                })
            return {'location': location, 'forecast': forecast_data}
        else:
            # For real API forecast, you'd use OpenWeatherMap's 5-day / 3-hour forecast API
            # This is a placeholder for real API integration
            logger.warning("Forecast not fully implemented for non-demo mode with real API.")
            return {"error": "Forecast API integration pending."}


# Utility functions for integration with app.py (can be removed if app.py calls WeatherService directly)
def get_current_weather_data(location: str = "Vellore, Tamil Nadu, IN") -> Dict:
    """Get current weather data for integration with main app"""
    weather_service = WeatherService()
    return weather_service.get_current_weather(location)


def get_forecast_data(location: str = "Vellore, Tamil Nadu, IN", days: int = 5) -> Dict:
    """Get forecast data for integration with main app"""
    weather_service = WeatherService()
    return weather_service.get_weather_forecast(location, days)


if __name__ == "__main__":
    # Test the weather service
    print("Testing Weather Service...")

    # Test with API key (replace with your actual key if you want to test real data)
    # weather_service = WeatherService(api_key="YOUR_OPENWEATHERMAP_API_KEY")
    # Test in demo mode
    weather_service = WeatherService(api_key=None) # Set to None for demo mode

    print("\n--- Current Weather (Demo Mode) ---")
    current = weather_service.get_current_weather("Hyderabad, Telangana, IN")
    print(f"Location: {current.get('location', 'N/A')}")
    print(f"Temperature: {current.get('temperature', 'N/A')}°C, Conditions: {current.get('conditions', 'N/A')}")
    print(f"Humidity: {current.get('humidity', 'N/A')}%, Wind: {current.get('wind_speed', 'N/A')} km/h, Rainfall: {current.get('rainfall', 'N/A')}mm")
    print("Alerts:")
    for alert in current.get('alerts', []):
        print(f"  [{alert['type'].upper()}] {alert['message']} {alert['icon']}")

    print("\n--- 5-Day Forecast (Demo Mode) ---")
    forecast = weather_service.get_weather_forecast("Hyderabad, Telangana, IN", days=5)
    if forecast and 'forecast' in forecast:
        for day_data in forecast['forecast']:
            print(f"Date: {day_data['date']}, Min/Max Temp: {day_data['min_temp']}°C/{day_data['max_temp']}°C, Conditions: {day_data['conditions']}, Rainfall: {day_data['rainfall']}mm")
    else:
        print(forecast.get('error', 'No forecast data available.'))

    # Example of how alerts are generated internally
    test_data_high_temp = {'temperature': 42, 'humidity': 60, 'wind_speed': 15, 'rainfall': 0}
    test_alerts = weather_service._check_weather_alerts(test_data_high_temp)
    print("\n--- Test Alerts (High Temp) ---")
    for alert in test_alerts:
        print(f"  [{alert['type'].upper()}] {alert['message']} {alert['icon']}")

    test_data_heavy_rain = {'temperature': 25, 'humidity': 90, 'wind_speed': 10, 'rainfall': 60}
    test_alerts_rain = weather_service._check_weather_alerts(test_data_heavy_rain)
    print("\n--- Test Alerts (Heavy Rain) ---")
    for alert in test_alerts_rain:
        print(f"  [{alert['type'].upper()}] {alert['message']} {alert['icon']}")