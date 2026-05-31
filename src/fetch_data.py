import os
import requests
from datetime import datetime
from utils import get_mongo_client

def fetch_raw_aqi(city, api_key):
    url = f"https://api.waqi.info/feed/{city}/?token={api_key}"
    res = requests.get(url, timeout=10)
    res.raise_for_status()
    data = res.json()
    if data.get("status") != "ok":
        raise ValueError(f"API Error: {data.get('error', 'Unknown')}")
    return data["data"]

def parse_to_dict(raw, city):
    iaqi = raw.get("iaqi", {})
    weather = raw.get("weather", {})
    time_info = raw.get("time", {})
    
    return {
        "city": city,
        "timestamp": time_info.get("iso", datetime.now().isoformat()),
        "aqi": raw.get("aqi"),
        "dominant_pollutant": raw.get("dominentpol"),
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
        "temp": weather.get("t", {}).get("v"),
        "humidity": weather.get("h", {}).get("v"),
        "wind_speed": weather.get("w", {}).get("v")
    }

if __name__ == "__main__":
    city = os.getenv("CITY", "Delhi")
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("Please set API_KEY in .env")

    print(f"Fetching AQI data for {city}...")
    raw = fetch_raw_aqi(city, api_key)
    doc = parse_to_dict(raw, city)

    db = get_mongo_client()
    result = db.raw_aqi.insert_one(doc)
    print(f"Data saved to MongoDB! Document ID: {result.inserted_id}")