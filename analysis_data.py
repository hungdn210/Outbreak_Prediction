import xarray as xr
import os
import data_repository
import pandas as pd
import re
import glob
from collections import defaultdict
from functools import reduce
import numpy as np
from math import radians, cos, sin, asin, sqrt
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime
from matplotlib.dates import date2num

def extract_year_from_nc_filename(filename):
    match = re.search(r'_(\d{4})\d{2}-\d{4}\d{2}\.nc$', filename)
    if match:
        return match.group(1)
    return None

def export_all_variables_to_csv(variables_list):
    for variable in variables_list:
        output_dir = os.path.join(
            os.path.dirname(__file__),
            data_repository.DATA_LOCATION,
            'data_csv_daily',
            variable
        )
        os.makedirs(output_dir, exist_ok=True)

        nc_files = glob.glob(os.path.join(os.path.dirname(__file__), data_repository.DATA_LOCATION, 'data_nc_daily', variable,'*.nc'))
        for file in nc_files:
            ds = xr.open_dataset(file, engine='h5netcdf')
            year = extract_year_from_nc_filename(file)
            print(year)

            print(f"{variable} contains variables: {list(ds.data_vars)}")

            if variable not in ds.data_vars:
                print(f"Variable {variable} not found in file!")
                continue

            data = ds[variable]

            if not data.dims or "time" not in data.dims:
                print(f"Skipping: {variable} has no time dimension")
                continue

            try:
                # Stack only the spatial dimensions into one "grid" level
                df = data.stack(grid=("projection_y_coordinate", "projection_x_coordinate")).to_dataframe()

                print(f"Before reset_index - index: {df.index.names}")
                print(f"Columns before reset: {df.columns.tolist()}")

                # Reset just 'time' to avoid index conflict
                df = df.reset_index(level='time')
                print(f"After reset_index - columns: {df.columns.tolist()}")

                # Remove duplicate columns if any
                df = df.loc[:, ~df.columns.duplicated()]

                # Drop rows with any missing values
                df = df.dropna()

                output_path = os.path.join(output_dir, f"{year}.csv")
                df.to_csv(output_path, index=False)
                print(f"Saved: {output_path}")

            except Exception as e:
                print(f"Failed to export {variable}: {type(e).__name__}: {e}")

def delete_unnecessary_columns_in_csv(variable_list):
    for variable in variable_list:
        input_dir = os.path.join(os.path.dirname(__file__),data_repository.DATA_LOCATION,data_repository.DATA_LOCATION_CSVvariable)
        csv_files = glob.glob(os.path.join(input_dir, '*.nc'))
        for file in csv_files:
            df = pd.read_csv(file)
            # get the year out of the file name
            base = os.path.basename(file)
            name, ext = os.path.splitext(base)
            if variable == 'groundfrost':
                df = df[["time", 'clim_season',"latitude", "longitude", "groundfrost"]]
            
def extract_year_in_full_date_from_nc_filename(filename):
    # Assumes format: 'rainfall_hadukgrid_uk_12km_day_YYYYMMDD-YYYYMMDD.nc'
    return os.path.basename(filename).split("_")[-1].split("-")[0][:4]

def extract_month_from_nc_filename(filename):
    basename = os.path.basename(filename)
    start_date_str = basename.split("_")[-1].split("-")[0]
    return int(start_date_str[4:6])

def export_all_variables_to_yearly_csv(variables_list):
    for variable in variables_list:
        input_dir = os.path.join(os.path.dirname(__file__), data_repository.DATA_LOCATION, data_repository.DATA_LOCATION_NC_DAILY, variable)
        output_dir = os.path.join(os.path.dirname(__file__), data_repository.DATA_LOCATION, data_repository.DATA_LOCATION_CSV_DAILY, variable)
        os.makedirs(output_dir, exist_ok=True)

        nc_files = glob.glob(os.path.join(input_dir, "*.nc"))
        nc_files_by_year = {}

        # Group files by year
        for file in nc_files:
            year = extract_year_in_full_date_from_nc_filename(file)
            nc_files_by_year.setdefault(year, []).append(file)

        for year, files in nc_files_by_year.items():
            print(f"\nCombining {len(files)} files for {variable} - Year: {year}")
            yearly_df_list = []

            # Check for missing months
            found_months = set()
            for file in files:
                found_months.add(extract_month_from_nc_filename(file))
            expected_months = set(range(1, 13))
            missing = expected_months - found_months
            if missing:
                missing_str = ", ".join(f"{m:02d}" for m in sorted(missing))
                print(f"Warning: Missing month(s) for {variable} in {year}: {missing_str}")

            # Load and combine monthly data
            for file in files:
                try:
                    ds = xr.open_dataset(file, engine='h5netcdf')
                    if variable not in ds.data_vars:
                        print(f"Variable '{variable}' not found in {file}")
                        continue

                    data = ds[variable]

                    if "time" not in data.dims:
                        print(f"Skipping {file} — no time dimension.")
                        continue

                    df = data.stack(grid=("projection_y_coordinate", "projection_x_coordinate")).to_dataframe()
                    df = df.reset_index(level='time')
                    df = df.loc[:, ~df.columns.duplicated()]
                    df = df.dropna()
                    yearly_df_list.append(df)

                except Exception as e:
                    print(f"Failed to process {file}: {type(e).__name__}: {e}")

            if yearly_df_list:
                full_df = pd.concat(yearly_df_list)
                output_path = os.path.join(output_dir, f"{year}.csv")
                full_df.to_csv(output_path, index=False)
                print(f"Saved yearly CSV: {output_path}")
            else:
                print(f"No data collected for {variable} in {year}")

def restructure_data_by_latlon(base_dir='dataset/daily_data_csv', output_dir='dataset/grid_point_timeseries'):
    os.makedirs(output_dir, exist_ok=True)
    
    variable_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    print(f"Detected variables: {variable_dirs}")

    grid_data = defaultdict(lambda: {})  # {(lat, lon): {variable: df}}

    for variable in variable_dirs:
        var_dir = os.path.join(base_dir, variable)
        csv_files = sorted(glob.glob(os.path.join(var_dir, "*.csv")))

        for file in csv_files:
            print(f"Reading {file}...")
            df = pd.read_csv(file)
            if df.empty:
                continue

            for (lat, lon), group in df.groupby(["latitude", "longitude"]):
                if variable not in group.columns:
                    print(f"Skipping: column '{variable}' not found in {file}")
                    continue

                sub_df = group[["time", variable]].copy()
                sub_df["time"] = pd.to_datetime(sub_df["time"])
                sub_df.set_index("time", inplace=True)
                sub_df.sort_index(inplace=True)

                # Combine with previous data for this variable if exists
                if variable in grid_data[(lat, lon)]:
                    grid_data[(lat, lon)][variable] = pd.concat([
                        grid_data[(lat, lon)][variable],
                        sub_df
                    ])
                else:
                    grid_data[(lat, lon)][variable] = sub_df

    # Merge per (lat, lon)
    for (lat, lon), var_df_dict in grid_data.items():
        if not var_df_dict:
            continue

        # Ensure no duplicates in each variable DataFrame
        cleaned_dfs = []
        for var_name, df in var_df_dict.items():
            df = df[~df.index.duplicated(keep='first')]
            df.columns = [var_name]  # reset clean column name
            cleaned_dfs.append(df)

        # Merge all variables on time
        merged_df = reduce(
            lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how="outer"),
            cleaned_dfs
        )

        merged_df.reset_index(inplace=True)

        filename = f"{lat:.5f}_{lon:.5f}.csv".replace(" ", "").replace("-", "m")
        path = os.path.join(output_dir, filename)
        merged_df.to_csv(path, index=False)

        print(f"Saved: {filename}")
        print("Example row:")
        print(merged_df.head(1).to_string(index=False))

# Approximate radius (in km) of Earth for haversine formula
EARTH_RADIUS_KM = 6371

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    return EARTH_RADIUS_KM * c

def find_all_main_places():
    input_csv = os.path.join("dataset", "data_BTV_cases", "2008And2024.csv")
    df = pd.read_csv(input_csv, encoding="ISO-8859-1")

    df.columns = df.columns.str.strip()
    df['latitude'] = df['latitude'].astype(str).str.strip().astype(float)
    df['longitude'] = df['longitude'].astype(str).str.strip().astype(float)

    grid_dir = os.path.join("dataset", "grid_point_date")
    grid_files = [f for f in os.listdir(grid_dir) if f.endswith('.csv')]

    grid_coords = []
    for file in grid_files:
        try:
            parts = file.replace(".csv", "").split("_")
            lat = float(parts[0])
            lon = float(parts[1].replace("m", "-"))
            grid_coords.append((lat, lon, file))
        except Exception as e:
            continue

    main_point_list = defaultdict(list)

    for _, row in df.iterrows():
        lat0, lon0 = row['latitude'], row['longitude']
        for grid_lat, grid_lon, filename in grid_coords:
            if haversine(lat0, lon0, grid_lat, grid_lon) <= 6:
                key = f"{grid_lat:.5f}_{grid_lon:.5f}".replace("-", "m")
                main_point_list[key].append({
                    "location": row.get("location", ""),
                    "latitude": lat0,
                    "longitude": lon0,
                    "startedOn": row.get("startedOn", ""),
                    "endedOn": row.get("endedOn", ""),
                    "Susceptible": row.get("Susceptible", None),
                    "Cases": row.get("Cases", None),
                    "Killed and disposed of": row.get("Killed and disposed of", None),
                    "Deaths": row.get("Deaths", None),
                    "Serotype": row.get("Serotype", None)
                })
                break  # Assign to the first matching grid only

    output_path = "dataset/main_point_list.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dict(main_point_list), f, indent=4)

    return output_path

def draw_charts_from_main_points(
    category='tasmin',
    json_path='dataset/main_point_list.json',
    grid_dir='dataset/grid_point_date'
):
    tmp = 0
    with open(json_path, 'r') as f:
        main_points = json.load(f)

    for key, records in main_points.items():
        file_path = os.path.join(grid_dir, f"{key}.csv")
        if not os.path.exists(file_path):
            print(f"Missing: {file_path}")
            continue

        df = pd.read_csv(file_path)
        if "time" not in df.columns or category not in df.columns:
            continue

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(df["time"], df[category], label=category, color="#aaa", zorder=1)

        max_val = df[category].max()

        for record in records:
            try:
                start = pd.to_datetime(record.get("startedOn", ""))
                end = pd.to_datetime(record.get("endedOn", ""))
                height = record.get("Cases", 0)
                height = height * 3

                if pd.isna(start) or pd.isna(end) or height == 0:
                    continue

                width = (end - start).days
                rect = Rectangle(
                    (date2num(start), 0),         # Bottom-left corner (x, y)
                    width,                        # Width (days)
                    height,                       # Height (cases)
                    facecolor='red',
                    edgecolor='black',
                    alpha=0.4,
                    zorder=2
                )
                ax.add_patch(rect)

            except Exception as e:
                print(f"Error drawing rectangle: {e}")

        ax.set_title(f"{category} Time Series - {key}")
        ax.set_xlabel("Date")
        ax.set_ylabel(f"{category} (unit)")
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        plt.show()

        # Only show one for now; remove break if you want to plot all
        tmp = tmp + 1
        if tmp == 3:
            break

# Example usage:
#export_all_variables_to_csv(list([data_repository.TOTAL_PRECIPITATION]))
#export_all_variables_to_yearly_csv(list([data_repository.NEAR_SURFACE_TEMPERATURE_MIN]))
#restructure_data_by_latlon()
#find_all_main_places()
draw_charts_from_main_points()
