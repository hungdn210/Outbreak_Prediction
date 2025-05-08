# General Definition
DATA_LOCATION = 'data'
# Variables Definition
DAYS_WITH_GROUND_FROST = 'groundfrost'  # Number of days with ground frost
RELATIVE_HUMIDITY_SURFACE = 'hurs'      # Relative humidity at the surface level (%)
SEA_LEVEL_PRESSURE = 'psl'              # Sea level pressure (hPa)
POTENTIAL_EVAPOTRANSPIRATION = 'pv'     # Potential evapotranspiration (mm)
TOTAL_PRECIPITATION = 'rainfall'        # Total precipitation including rain, snow, sleet, hail (mm)
SURFACE_WIND_SPEED = 'sfcWind'          # Surface wind speed (m/s)
DAYS_WITH_SNOW_LYING = 'snowLying'      # Number of days with snow lying on the ground
TOTAL_SUNSHINE_DURATION = 'sun'         # Total sunshine duration (hours)
NEAR_SURFACE_TEMPERATURE = 'tas'        # Near-surface air temperature (°C)

VARIABLE_METADATA = {
    DAYS_WITH_GROUND_FROST: {
        'name': 'groundfrost',
        'description': 'Number of days with ground frost (minimum air temperature below freezing).',
        'unit': 'days',
        'start_date': '1991-01-01',
        'end_date': '2000-12-31'
    },
    RELATIVE_HUMIDITY_SURFACE: {
        'name': 'hurs',
        'description': 'Relative humidity at the surface level.',
        'unit': '%',
        'start_date': '1991-01-01',
        'end_date': '2000-12-31'
    },
    SEA_LEVEL_PRESSURE: {
        'name': 'psl',
        'description': 'Atmospheric pressure reduced to sea level.',
        'unit': 'hPa',
        'start_date': '1991-01-01',
        'end_date': '2000-12-31'
    },
    POTENTIAL_EVAPOTRANSPIRATION: {
        'name': 'pv',
        'description': 'The amount of evaporation that would occur with unlimited water supply.',
        'unit': 'mm',
        'start_date': '1991-01-01',
        'end_date': '2000-12-31'
    },
    TOTAL_PRECIPITATION: {
        'name': 'rainfall',
        'description': 'Total daily precipitation including rain, snow, sleet, and hail.',
        'unit': 'mm',
        'start_date': '1991-01-01',
        'end_date': '2000-12-31'
    },
    SURFACE_WIND_SPEED: {
        'name': 'sfcWind',
        'description': 'Wind speed measured at the surface level.',
        'unit': 'm/s',
        'start_date': '1991-01-01',
        'end_date': '2000-12-31'
    },
    DAYS_WITH_SNOW_LYING: {
        'name': 'snowLying',
        'description': 'Number of days with snow lying on the ground.',
        'unit': 'days',
        'start_date': '1991-01-01',
        'end_date': '2000-12-31'
    },
    TOTAL_SUNSHINE_DURATION: {
        'name': 'sun',
        'description': 'Total duration of sunshine during the day.',
        'unit': 'hours',
        'start_date': '1991-01-01',
        'end_date': '2000-12-31'
    },
    NEAR_SURFACE_TEMPERATURE: {
        'name': 'tas',
        'description': 'Daily mean near-surface air temperature.',
        'unit': '°C',
        'start_date': '1991-01-01',
        'end_date': '2000-12-31'
    }
}