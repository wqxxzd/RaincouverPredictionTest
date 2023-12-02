import openmeteo_requests
import requests_cache
import pandas as pd
import os

from retry_requests import retry
from datetime import datetime, timedelta

def get_vancouver_data(url, start_date, end_date, write_to = "", create_csv = False):
    """
    Creates a new DataFrame with 18 columns, containing weather observations for each date between 
    the start and end dates in Vancouver. Data is extracted via API from  Open-Meteo’s Historical Weather 
    API. Each row in the dataset includes weather measurement statistics in a day. 

    Parameters:
    ----------
    url : str
        A string url that serves as the API endpoint to get the data from
    start_date : str
        A string in YYYY-MM-DD format (e.g. "1990-01-01") that the weather API will start extracting from.
    end_date : str
        A string in YYYY-MM-DD format (e.g. "2000-01-01") that the weather API will conclude the query.
    write_to : str
        A string path for the csv file to be stored.
    create_csv: bool
        A boolean. If true, a csv file will be created in data folder, populated with weather data. False by default. 

    Returns:
    -------
    pandas.DataFrame
        A DatetimeIndex DataFrame with 18 columns, containing weather observations for each date 
        between the start and end dates. 
        
    Examples:
    --------
    >>> precipit_df = get_vancouver_data(start_date, end_date, create_csv=True)

    """

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    VAN_LAT = 49.2497
    VAN_LONG = -123.1193
    START_DATE = start_date # default to "1990-01-01"
    END_DATE = end_date # default to (datetime.now() - timedelta(days = 7)).strftime('%Y-%m-%d')
    RETRIEVE_COLS = ["weather_code", "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean", 
                     "apparent_temperature_max", "apparent_temperature_min", "apparent_temperature_mean", 
                     "sunrise", "sunset", "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours", 
                     "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant", "shortwave_radiation_sum",
                     "et0_fao_evapotranspiration"]
    
    
    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    #url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
    	"latitude": VAN_LAT,
    	"longitude": VAN_LONG,
    	"start_date": START_DATE,
    	"end_date": END_DATE,
    	"daily": RETRIEVE_COLS,
        "timezone": "auto"
    }
    responses = openmeteo.weather_api(url, params=params)
    
    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°E {response.Longitude()}°N")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")
    
    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_weather_code = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_max = daily.Variables(1).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(2).ValuesAsNumpy()
    daily_temperature_2m_mean = daily.Variables(3).ValuesAsNumpy()
    daily_apparent_temperature_max = daily.Variables(4).ValuesAsNumpy()
    daily_apparent_temperature_min = daily.Variables(5).ValuesAsNumpy()
    daily_apparent_temperature_mean = daily.Variables(6).ValuesAsNumpy()
    daily_sunrise = daily.Variables(7).ValuesAsNumpy()
    daily_sunset = daily.Variables(8).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(9).ValuesAsNumpy()
    daily_rain_sum = daily.Variables(10).ValuesAsNumpy()
    daily_snowfall_sum = daily.Variables(11).ValuesAsNumpy()
    daily_precipitation_hours = daily.Variables(12).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(13).ValuesAsNumpy()
    daily_wind_gusts_10m_max = daily.Variables(14).ValuesAsNumpy()
    daily_wind_direction_10m_dominant = daily.Variables(15).ValuesAsNumpy()
    daily_shortwave_radiation_sum = daily.Variables(16).ValuesAsNumpy()
    daily_et0_fao_evapotranspiration = daily.Variables(17).ValuesAsNumpy()
    
    daily_data = {"date": pd.date_range(
    	start = pd.to_datetime(daily.Time(), unit = "s").strftime('%Y-%m-%d'),
    	end = pd.to_datetime(daily.TimeEnd(), unit = "s").strftime('%Y-%m-%d'),
    	freq = pd.Timedelta(days = 1),
    	inclusive = "left"
    )}
    daily_data["weather_code"] = daily_weather_code
    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_min"] = daily_temperature_2m_min
    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data["apparent_temperature_max"] = daily_apparent_temperature_max
    daily_data["apparent_temperature_min"] = daily_apparent_temperature_min
    daily_data["apparent_temperature_mean"] = daily_apparent_temperature_mean
    daily_data["sunrise"] = daily_sunrise
    daily_data["sunset"] = daily_sunset
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["rain_sum"] = daily_rain_sum
    daily_data["snowfall_sum"] = daily_snowfall_sum
    daily_data["precipitation_hours"] = daily_precipitation_hours
    daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
    daily_data["wind_gusts_10m_max"] = daily_wind_gusts_10m_max
    daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant
    daily_data["shortwave_radiation_sum"] = daily_shortwave_radiation_sum
    daily_data["et0_fao_evapotranspiration"] = daily_et0_fao_evapotranspiration
    
    df_van_weather = pd.DataFrame(data = daily_data)
    df_van_weather = df_van_weather.set_index('date')

    if create_csv == True:  # Publish to CSV file if create_csv parameter is True

        # write_to path transforming
        if write_to != '':
            write_to = write_to if write_to[-1] == '/' else write_to + '/'
        
        # Check write_to path existence
        if not os.path.exists(write_to):
            os.mkdir(write_to)

        full_path = os.path.join(write_to, f'van_weather_{start_date}_{end_date}.csv')

        df_van_weather.to_csv(full_path)
        print(f'published to {full_path}')
        
    return df_van_weather