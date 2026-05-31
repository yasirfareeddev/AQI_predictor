import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils import get_mongo_client

def generate_historical_data(city, days=30):
    np.random.seed(42)
    start = datetime.now() - timedelta(days=days)
    timestamps = [start + timedelta(hours=h) for h in range(days*24)]
    
    # Simulate realistic AQI & pollutant patterns (diurnal cycle + noise)
    hours = np.array([t.hour for t in timestamps])
    base_aqi = 80 + 30 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 10, len(timestamps))
    base_aqi = np.clip(base_aqi, 20, 300).astype(int)
    
    records = []
    for i, ts in enumerate(timestamps):
        aqi = int(base_aqi[i])
        pm25 = max(10, aqi * 0.4 + np.random.normal(0, 5))
        pm10 = max(15, pm25 * 1.8 + np.random.normal(0, 8))
        o3 = max(5, 40 + 20 * np.sin(2*np.pi*hours[i]/24) + np.random.normal(0, 5))
        no2 = max(2, 15 + 10 * np.sin(2*np.pi*(hours[i]-6)/24) + np.random.normal(0, 3))
        
        records.append({
            "city": city,
            "timestamp": ts.isoformat(),
            "aqi": aqi,
            "dominant_pollutant": "pm25" if pm25 > pm10 else "pm10",
            "pm25": round(pm25, 2),
            "pm10": round(pm10, 2),
            "o3": round(o3, 2),
            "no2": round(no2, 2),
            "so2": round(np.random.uniform(2, 15), 2),
            "co": round(np.random.uniform(200, 1000), 2),
            "temp": round(np.random.normal(25, 5), 1),
            "humidity": round(np.random.uniform(40, 80), 1),
            "wind_speed": round(np.random.uniform(2, 15), 1)
        })
    return records

if __name__ == "__main__":
    city = os.getenv("CITY", "Islamabad")
    print(f"Backfilling {city} with 30 days of hourly data...")
    docs = generate_historical_data(city)
    
    db = get_mongo_client()
    db.raw_aqi.insert_many(docs)
    print(f"Inserted {len(docs)} documents into raw_aqi collection.")