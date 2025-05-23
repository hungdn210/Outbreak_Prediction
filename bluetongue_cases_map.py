import pandas as pd
import folium
from folium.plugins import MarkerCluster
import os
import data_repository
import glob

# Sample dataset 
input_dir = os.path.join(os.path.dirname(__file__), data_repository.DATA_LOCATION, data_repository.DATA_BTV_CASES)
csv_2008_file = os.path.join(input_dir, "2008.csv")
csv_2024_file = os.path.join(input_dir, '2024.csv')
csv_2025_file = os.path.join(input_dir, "2025-ongoing.csv")

df_2008 = pd.read_csv(csv_2008_file, encoding="ISO-8859-1")
df_2008.columns = df_2008.columns.str.strip()
df_2024 = pd.read_csv(csv_2024_file, encoding="ISO-8859-1")
df_2024.columns = df_2024.columns.str.strip()
df_2025 = pd.read_csv(csv_2025_file, encoding="ISO-8859-1")
df_2025.columns = df_2025.columns.str.strip()

# Create the base map centered over the UK
m = folium.Map(location=[52.0, 0.5], zoom_start=6)

case_size = 300
offset = 0.01  # about 1km

for _, row in df_2008.iterrows():
    lat = float(row['latitude'])
    lon = float(row['longitude'])
    cases = float(row['Cases'])

    # Radius circle based on number of cases
    folium.Circle(
        location=[lat, lon],
        radius=cases * case_size, 
        color="darkgreen",
        fill=True,
        fill_color="darkgreen",
        fill_opacity=0.3,
    ).add_to(m)

    # Standard icon marker
    folium.Marker(
        location=[lat, lon],
        icon=folium.Icon(color="darkgreen"),
        popup=(f"{row['location']}<br>Cases: {cases}<br>Start Date: {row['startedOn']}<br>End Date: {row['endedOn']}")
    ).add_to(m)

for _, row in df_2024.iterrows():
    lat = float(row['latitude']) + offset
    lon = float(row['longitude']) + offset
    cases = float(row['Cases'])

    # Radius circle based on number of cases
    folium.Circle(
        location=[lat, lon],
        radius=cases * case_size,  
        color="orange",
        fill=True,
        fill_color="orange",
        fill_opacity=0.3,
    ).add_to(m)

    # Standard icon marker
    folium.Marker(
        location=[lat, lon],
        icon=folium.Icon(color="orange"),
        popup=(f"{row['location']}<br>Cases: {cases}<br>Start Date: {row['startedOn']}<br>End Date: {row['endedOn']}")
    ).add_to(m)

for _, row in df_2025.iterrows():
    lat = float(row['latitude']) - offset
    lon = float(row['longitude']) - offset
    cases = float(row['Cases'])

    # Radius circle based on number of cases
    folium.Circle(
        location=[lat, lon],
        radius=cases * case_size, 
        color="red",
        fill=True,
        fill_color="red",
        fill_opacity=0.3,
    ).add_to(m)

    # Standard icon marker
    folium.Marker(
        location=[lat, lon],
        icon=folium.Icon(color="red"),
        popup=(f"{row['location']}<br>Cases: {cases}<br>Start Date: {row['startedOn']}<br>End Date: {row['endedOn']}")
    ).add_to(m)
# Save map to file
m.save("bluetongue_map.html")
print("Map saved as 'bluetongue_map.html'")