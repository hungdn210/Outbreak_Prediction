import xarray as xr
import os
import data_repository

def preview_nc_file(variables_list, top_rows=5, bottom_rows=5):
    for variable in variables_list:
        ds = xr.open_dataset(os.path.join(os.path.dirname(__file__), data_repository.DATA_LOCATION, f'{variable}.nc'))
        
        column_names = list(ds.data_vars)
        print(variable, column_names)

preview_nc_file(data_repository.VARIABLE_METADATA.items)
    