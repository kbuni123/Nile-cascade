# weather_integration.py
import requests
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class WeatherData:
    """Standardized weather data structure"""
    date: datetime
    temperature_min: float
    temperature_max: float
    precipitation: float
    humidity: float
    evapotranspiration: Optional[float] = None
    wind_speed: Optional[float] = None
    location: str = ""

class FreeWeatherDataManager:
    """Weather data manager using only free sources and historical data"""
    
    def __init__(self):
        # Nile basin key locations
        self.locations = {
            'addis_ababa': {'lat': 9.0320, 'lon': 38.7469, 'name': 'Addis Ababa, Ethiopia'},
            'khartoum': {'lat': 15.5527, 'lon': 32.5599, 'name': 'Khartoum, Sudan'},
            'cairo': {'lat': 30.0444, 'lon': 31.2357, 'name': 'Cairo, Egypt'},
            'bahir_dar': {'lat': 11.6000, 'lon': 37.3833, 'name': 'Bahir Dar, Ethiopia'},
            'wad_medani': {'lat': 14.4010, 'lon': 33.5158, 'name': 'Wad Medani, Sudan'},
            'aswan': {'lat': 24.0889, 'lon': 32.8998, 'name': 'Aswan, Egypt'},
            'jinja': {'lat': 0.4244, 'lon': 33.2042, 'name': 'Jinja, Uganda'}
        }
        
        # Historical monthly climate data (30-year averages from various free sources)
        # Data compiled from World Bank Climate Portal, NOAA, and historical records
        self.historical_climate = {
            'addis_ababa': {
                1: {'temp_min': 6, 'temp_max': 23, 'precip': 13, 'humidity': 52},
                2: {'temp_min': 8, 'temp_max': 24, 'precip': 30, 'humidity': 48},
                3: {'temp_min': 10, 'temp_max': 25, 'precip': 60, 'humidity': 45},
                4: {'temp_min': 11, 'temp_max': 24, 'precip': 80, 'humidity': 55},
                5: {'temp_min': 11, 'temp_max': 25, 'precip': 85, 'humidity': 53},
                6: {'temp_min': 10, 'temp_max': 23, 'precip': 140, 'humidity': 68},
                7: {'temp_min': 10, 'temp_max': 20, 'precip': 280, 'humidity': 79},
                8: {'temp_min': 10, 'temp_max': 20, 'precip': 290, 'humidity': 79},
                9: {'temp_min': 9, 'temp_max': 21, 'precip': 150, 'humidity': 72},
                10: {'temp_min': 7, 'temp_max': 22, 'precip': 25, 'humidity': 58},
                11: {'temp_min': 6, 'temp_max': 22, 'precip': 7, 'humidity': 56},
                12: {'temp_min': 5, 'temp_max': 22, 'precip': 7, 'humidity': 54}
            },
            'khartoum': {
                1: {'temp_min': 16, 'temp_max': 31, 'precip': 0, 'humidity': 29},
                2: {'temp_min': 18, 'temp_max': 33, 'precip': 0, 'humidity': 23},
                3: {'temp_min': 21, 'temp_max': 37, 'precip': 0.1, 'humidity': 17},
                4: {'temp_min': 24, 'temp_max': 40, 'precip': 1, 'humidity': 16},
                5: {'temp_min': 27, 'temp_max': 42, 'precip': 4, 'humidity': 21},
                6: {'temp_min': 27, 'temp_max': 41, 'precip': 7, 'humidity': 32},
                7: {'temp_min': 25, 'temp_max': 38, 'precip': 30, 'humidity': 47},
                8: {'temp_min': 25, 'temp_max': 37, 'precip': 49, 'humidity': 52},
                9: {'temp_min': 26, 'temp_max': 39, 'precip': 27, 'humidity': 41},
                10: {'temp_min': 25, 'temp_max': 39, 'precip': 8, 'humidity': 31},
                11: {'temp_min': 20, 'temp_max': 35, 'precip': 1, 'humidity': 30},
                12: {'temp_min': 17, 'temp_max': 32, 'precip': 0, 'humidity': 32}
            },
            'cairo': {
                1: {'temp_min': 9, 'temp_max': 19, 'precip': 5, 'humidity': 59},
                2: {'temp_min': 10, 'temp_max': 21, 'precip': 4, 'humidity': 54},
                3: {'temp_min': 12, 'temp_max': 24, 'precip': 3, 'humidity': 52},
                4: {'temp_min': 14, 'temp_max': 28, 'precip': 1, 'humidity': 47},
                5: {'temp_min': 18, 'temp_max': 32, 'precip': 0.5, 'humidity': 46},
                6: {'temp_min': 20, 'temp_max': 34, 'precip': 0, 'humidity': 49},
                7: {'temp_min': 22, 'temp_max': 35, 'precip': 0, 'humidity': 58},
                8: {'temp_min': 22, 'temp_max': 35, 'precip': 0, 'humidity': 61},
                9: {'temp_min': 20, 'temp_max': 33, 'precip': 0, 'humidity': 60},
                10: {'temp_min': 17, 'temp_max': 30, 'precip': 1, 'humidity': 59},
                11: {'temp_min': 14, 'temp_max': 25, 'precip': 3, 'humidity': 61},
                12: {'temp_min': 10, 'temp_max': 21, 'precip': 6, 'humidity': 61}
            },
            'bahir_dar': {
                1: {'temp_min': 9, 'temp_max': 27, 'precip': 5, 'humidity': 51},
                2: {'temp_min': 10, 'temp_max': 29, 'precip': 7, 'humidity': 44},
                3: {'temp_min': 13, 'temp_max': 30, 'precip': 18, 'humidity': 42},
                4: {'temp_min': 14, 'temp_max': 30, 'precip': 34, 'humidity': 45},
                5: {'temp_min': 14, 'temp_max': 29, 'precip': 87, 'humidity': 56},
                6: {'temp_min': 14, 'temp_max': 26, 'precip': 174, 'humidity': 74},
                7: {'temp_min': 14, 'temp_max': 24, 'precip': 396, 'humidity': 83},
                8: {'temp_min': 14, 'temp_max': 24, 'precip': 389, 'humidity': 83},
                9: {'temp_min': 13, 'temp_max': 25, 'precip': 211, 'humidity': 78},
                10: {'temp_min': 12, 'temp_max': 26, 'precip': 89, 'humidity': 69},
                11: {'temp_min': 10, 'temp_max': 26, 'precip': 20, 'humidity': 61},
                12: {'temp_min': 9, 'temp_max': 26, 'precip': 6, 'humidity': 56}
            },
            'wad_medani': {
                1: {'temp_min': 15, 'temp_max': 33, 'precip': 0, 'humidity': 36},
                2: {'temp_min': 17, 'temp_max': 35, 'precip': 0, 'humidity': 29},
                3: {'temp_min': 20, 'temp_max': 38, 'precip': 0.5, 'humidity': 24},
                4: {'temp_min': 23, 'temp_max': 40, 'precip': 3, 'humidity': 23},
                5: {'temp_min': 25, 'temp_max': 41, 'precip': 12, 'humidity': 31},
                6: {'temp_min': 25, 'temp_max': 39, 'precip': 38, 'humidity': 45},
                7: {'temp_min': 24, 'temp_max': 35, 'precip': 95, 'humidity': 61},
                8: {'temp_min': 23, 'temp_max': 34, 'precip': 115, 'humidity': 66},
                9: {'temp_min': 24, 'temp_max': 36, 'precip': 53, 'humidity': 54},
                10: {'temp_min': 23, 'temp_max': 38, 'precip': 15, 'humidity': 41},
                11: {'temp_min': 19, 'temp_max': 36, 'precip': 1, 'humidity': 37},
                12: {'temp_min': 16, 'temp_max': 33, 'precip': 0, 'humidity': 38}
            },
            'aswan': {
                1: {'temp_min': 10, 'temp_max': 23, 'precip': 0, 'humidity': 40},
                2: {'temp_min': 12, 'temp_max': 26, 'precip': 0, 'humidity': 33},
                3: {'temp_min': 15, 'temp_max': 30, 'precip': 0, 'humidity': 26},
                4: {'temp_min': 20, 'temp_max': 35, 'precip': 0, 'humidity': 20},
                5: {'temp_min': 24, 'temp_max': 39, 'precip': 0, 'humidity': 17},
                6: {'temp_min': 26, 'temp_max': 41, 'precip': 0, 'humidity': 16},
                7: {'temp_min': 27, 'temp_max': 41, 'precip': 0, 'humidity': 18},
                8: {'temp_min': 27, 'temp_max': 41, 'precip': 0, 'humidity': 21},
                9: {'temp_min': 24, 'temp_max': 39, 'precip': 0, 'humidity': 26},
                10: {'temp_min': 20, 'temp_max': 35, 'precip': 0, 'humidity': 32},
                11: {'temp_min': 15, 'temp_max': 29, 'precip': 0, 'humidity': 38},
                12: {'temp_min': 11, 'temp_max': 24, 'precip': 0, 'humidity': 42}
            },
            'jinja': {
                1: {'temp_min': 18, 'temp_max': 28, 'precip': 58, 'humidity': 69},
                2: {'temp_min': 18, 'temp_max': 29, 'precip': 61, 'humidity': 66},
                3: {'temp_min': 18, 'temp_max': 28, 'precip': 132, 'humidity': 70},
                4: {'temp_min': 18, 'temp_max': 27, 'precip': 175, 'humidity': 75},
                5: {'temp_min': 18, 'temp_max': 27, 'precip': 147, 'humidity': 76},
                6: {'temp_min': 17, 'temp_max': 27, 'precip': 74, 'humidity': 73},
                7: {'temp_min': 17, 'temp_max': 27, 'precip': 65, 'humidity': 72},
                8: {'temp_min': 17, 'temp_max': 27, 'precip': 83, 'humidity': 73},
                9: {'temp_min': 17, 'temp_max': 28, 'precip': 91, 'humidity': 72},
                10: {'temp_min': 17, 'temp_max': 28, 'precip': 124, 'humidity': 73},
                11: {'temp_min': 17, 'temp_max': 27, 'precip': 122, 'humidity': 74},
                12: {'temp_min': 17, 'temp_max': 27, 'precip': 86, 'humidity': 72}
            }
        }
    
    def fetch_open_meteo_forecast(self, location: str, days: int = 14) -> List[WeatherData]:
        """Fetch FREE weather data from Open-Meteo API (no key required)"""
        try:
            loc_data = self.locations.get(location)
            if not loc_data:
                raise ValueError(f"Invalid location: {location}")
            
            # Open-Meteo API - completely free, no API key needed
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                'latitude': loc_data['lat'],
                'longitude': loc_data['lon'],
                'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,relative_humidity_2m_mean,windspeed_10m_max,et0_fao_evapotranspiration',
                'timezone': 'Africa/Cairo',
                'forecast_days': min(days, 16)  # Free tier supports up to 16 days
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            weather_data = []
            daily = data.get('daily', {})
            
            for i in range(len(daily.get('time', []))):
                weather_data.append(WeatherData(
                    date=datetime.strptime(daily['time'][i], '%Y-%m-%d'),
                    temperature_min=daily['temperature_2m_min'][i],
                    temperature_max=daily['temperature_2m_max'][i],
                    precipitation=daily['precipitation_sum'][i] or 0,
                    humidity=daily['relative_humidity_2m_mean'][i],
                    evapotranspiration=daily['et0_fao_evapotranspiration'][i] or self._calculate_eto(
                        (daily['temperature_2m_min'][i] + daily['temperature_2m_max'][i]) / 2,
                        daily['relative_humidity_2m_mean'][i],
                        daily['windspeed_10m_max'][i]
                    ),
                    wind_speed=daily['windspeed_10m_max'][i],
                    location=loc_data['name']
                ))
            
            logger.info(f"Successfully fetched Open-Meteo forecast for {location}")
            return weather_data
            
        except Exception as e:
            logger.error(f"Open-Meteo API error for {location}: {str(e)}")
            return []
    
    def fetch_wttr_in_data(self, location: str, days: int = 3) -> List[WeatherData]:
        """Fetch FREE weather data from wttr.in (no key required, max 3 days)"""
        try:
            loc_data = self.locations.get(location)
            if not loc_data:
                raise ValueError(f"Invalid location: {location}")
            
            # wttr.in API - free service, no key needed
            url = f"https://wttr.in/{loc_data['lat']},{loc_data['lon']}"
            params = {
                'format': 'j1',  # JSON format
                'days': min(days, 3)  # Max 3 days free
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            weather_data = []
            for day in data.get('weather', []):
                date = datetime.strptime(day['date'], '%Y-%m-%d')
                
                # Extract daily values
                temp_min = float(day['mintempC'])
                temp_max = float(day['maxtempC'])
                precip = float(day.get('totalSnow_cm', 0)) * 10 + sum(
                    float(hour.get('precipMM', 0)) for hour in day.get('hourly', [])
                ) / len(day.get('hourly', [1]))
                humidity = np.mean([float(h.get('humidity', 50)) for h in day.get('hourly', [])])
                wind = np.mean([float(h.get('windspeedKmph', 10)) for h in day.get('hourly', [])])
                
                weather_data.append(WeatherData(
                    date=date,
                    temperature_min=temp_min,
                    temperature_max=temp_max,
                    precipitation=precip,
                    humidity=humidity,
                    evapotranspiration=self._calculate_eto((temp_min + temp_max) / 2, humidity, wind / 3.6),
                    wind_speed=wind / 3.6,  # Convert km/h to m/s
                    location=loc_data['name']
                ))
            
            logger.info(f"Successfully fetched wttr.in data for {location}")
            return weather_data
            
        except Exception as e:
            logger.error(f"wttr.in API error for {location}: {str(e)}")
            return []
    
    def generate_seasonal_forecast(self, location: str, days: int = 30) -> List[WeatherData]:
        """Generate weather forecast based on historical seasonal patterns"""
        loc_data = self.locations.get(location)
        if not loc_data:
            # Use default climate if location not in database
            location = 'khartoum'  # Default to Khartoum climate
        
        # Get default climate data
        default_climate = self.historical_climate.get(location, self.historical_climate['khartoum'])
        
        weather_data = []
        current_date = datetime.now()
        
        for i in range(days):
            date = current_date + timedelta(days=i)
            month = date.month
            
            # Get monthly climate data
            monthly_climate = default_climate.get(month, default_climate[1])
            
            # Add daily variability to monthly averages
            temp_variation = np.random.normal(0, 2)  # ±2°C variation
            precip_factor = np.random.exponential(1)  # Exponential distribution for precipitation
            humidity_variation = np.random.normal(0, 5)  # ±5% humidity variation
            
            temp_min = monthly_climate['temp_min'] + temp_variation
            temp_max = monthly_climate['temp_max'] + temp_variation
            
            # Precipitation: use monthly total divided by 30, with random factor
            daily_precip = (monthly_climate['precip'] / 30) * precip_factor
            
            # Add occasional rain events (20% chance of significant rain in rainy months)
            if monthly_climate['precip'] > 50 and np.random.random() < 0.2:
                daily_precip *= np.random.uniform(3, 8)
            
            humidity = max(10, min(95, monthly_climate['humidity'] + humidity_variation))
            wind_speed = 2.5 + np.random.exponential(1)  # Base wind speed with variation
            
            weather_data.append(WeatherData(
                date=date,
                temperature_min=temp_min,
                temperature_max=temp_max,
                precipitation=max(0, daily_precip),
                humidity=humidity,
                evapotranspiration=self._calculate_eto((temp_min + temp_max) / 2, humidity, wind_speed),
                wind_speed=wind_speed,
                location=self.locations.get(location, {}).get('name', location)
            ))
        
        return weather_data
    
    def fetch_weather_data(self, location: str, days: int = 14) -> List[WeatherData]:
        """Fetch weather data using free sources with automatic fallback"""
        
        # Try Open-Meteo first (most reliable free source)
        weather_data = self.fetch_open_meteo_forecast(location, days)
        
        # If Open-Meteo fails and we need less than 3 days, try wttr.in
        if not weather_data and days <= 3:
            logger.warning(f"Open-Meteo failed for {location}, trying wttr.in...")
            weather_data = self.fetch_wttr_in_data(location, days)
        
        # Final fallback: use historical seasonal patterns
        if not weather_data:
            logger.warning(f"All free APIs failed for {location}, using historical seasonal patterns")
            weather_data = self.generate_seasonal_forecast(location, days)
        
        return weather_data
    
    def _calculate_eto(self, temp: float, humidity: float, wind_speed: float) -> float:
        """Calculate reference evapotranspiration using simplified Penman equation"""
        # Simplified ET0 calculation (mm/day)
        # Based on FAO-56 Penman-Monteith equation (simplified)
        
        # Saturation vapor pressure
        es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
        
        # Actual vapor pressure
        ea = es * (humidity / 100)
        
        # Vapor pressure deficit
        vpd = es - ea
        
        # Simplified ET0 (mm/day) - empirical formula
        et0 = 0.0023 * (temp + 17.8) * np.sqrt(abs(vpd)) * (0.5 + 0.54 * wind_speed)
        
        # Add temperature factor
        if temp > 30:
            et0 *= 1.1  # Increase ET in hot conditions
        elif temp < 10:
            et0 *= 0.7  # Decrease ET in cold conditions
        
        return max(0, min(15, et0))  # Cap between 0-15 mm/day
    
    def get_basin_forecast(self, days: int = 14, use_historical: bool = False) -> Dict[str, List[WeatherData]]:
        """Get weather forecast for all key locations in Nile basin"""
        basin_forecast = {}
        
        for location in self.locations.keys():
            try:
                if use_historical:
                    # Force use of historical patterns
                    basin_forecast[location] = self.generate_seasonal_forecast(location, days)
                else:
                    # Try free APIs first, fallback to historical
                    basin_forecast[location] = self.fetch_weather_data(location, days)
                    
            except Exception as e:
                logger.error(f"Failed to fetch weather data for {location}: {str(e)}")
                # Always provide some data using historical patterns
                basin_forecast[location] = self.generate_seasonal_forecast(location, days)
        
        return basin_forecast
    
    def get_historical_analysis(self, location: str, months: int = 12) -> pd.DataFrame:
        """Get historical climate analysis for planning"""
        if location not in self.locations:
            location = 'khartoum'  # Default
        
        climate_data = self.historical_climate.get(location, self.historical_climate['khartoum'])
        
        # Create DataFrame with monthly statistics
        monthly_stats = []
        for month in range(1, 13):
            month_data = climate_data.get(month, climate_data[1])
            monthly_stats.append({
                'Month': month,
                'Avg_Temp_Min': month_data['temp_min'],
                'Avg_Temp_Max': month_data['temp_max'],
                'Avg_Precipitation': month_data['precip'],
                'Avg_Humidity': month_data['humidity'],
                'Est_Evapotranspiration': self._calculate_eto(
                    (month_data['temp_min'] + month_data['temp_max']) / 2,
                    month_data['humidity'],
                    2.5  # Average wind speed
                )
            })
        
        return pd.DataFrame(monthly_stats)

# Usage example
weather_manager = FreeWeatherDataManager()

# Get forecast using free APIs
forecast_data = weather_manager.get_basin_forecast(days=14, use_historical=False)

# Or force historical patterns (always available, no API needed)
historical_forecast = weather_manager.get_basin_forecast(days=30, use_historical=True)

# Get historical analysis for planning
historical_stats = weather_manager.get_historical_analysis('bahir_dar', months=12)
print(historical_stats)