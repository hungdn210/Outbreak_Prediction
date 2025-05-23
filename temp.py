import requests
import pandas as pd

lat = 50.9129
lon = 0.1782
api_key = "6CBJM53GYBAHUV36UKD4ZK6DF"

url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}?unitGroup=metric&include=days&key={api_key}&contentType=json"

response = requests.get(url)

# Check for issues
if response.status_code != 200:
    print("❌ Request failed!")
    print("Status code:", response.status_code)
    print("Response text:", response.text)
else:
    data = response.json()
    df = pd.DataFrame(data['days'])
    print(df[['datetime', 'temp', 'precip', 'humidity', 'description']].head())
    df.to_csv("weather_data.csv", index=False)
    print("✅ Data saved to weather_data.csv")
