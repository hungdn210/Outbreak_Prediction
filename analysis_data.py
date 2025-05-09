import xarray as xr
import os
import data_repository
import pandas as pd
from tqdm import tqdm
 
def preview_nc_file(variables_list, top_rows=5, bottom_rows=5):
    print(variables_list)
    for variable in variables_list:
        ds = xr.open_dataset(os.path.join(os.path.dirname(__file__), data_repository.DATA_LOCATION, f'{variable}.nc'), engine="h5netcdf")
        column_names = list(ds.data_vars)
        print(variable, column_names)
 
def export_all_variables_to_csv(variables_list):
    for variable in variables_list:
        output_dir = os.path.join(os.path.dirname(__file__), data_repository.DATA_LOCATION, 'Original Data', variable)
        os.makedirs(output_dir, exist_ok=True)
        ds = xr.open_dataset(os.path.join(os.path.dirname(__file__), data_repository.DATA_LOCATION, f'{variable}.nc'), engine='h5netcdf')
        data = ds[variable]
        if not data.dims or "time" not in data.dims:
            print(f"Skipping metadata: {variable}")
            continue
 
        time_index = pd.to_datetime(data['time'].values)
        years = sorted(set(t.year for t in time_index))
        print(variable, years)
 
 
 
 
#preview_nc_file(list(data_repository.VARIABLE_METADATA.keys()))
#export_all_variables_to_csv(list([data_repository.NEAR_SURFACE_TEMPERATURE_MIN, data_repository.NEAR_SURFACE_TEMPERATURE_MAX]))
export_all_variables_to_csv(list(data_repository.VARIABLE_METADATA.keys()))