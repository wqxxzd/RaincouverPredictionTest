# Reference:
# https://raw.githubusercontent.com/ttimbers/breast_cancer_predictor_py/main/scripts/download_data.py
import click
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.get_api import get_vancouver_data

@click.command()
@click.option('--url', type=str, help="The url of the api endpoint")
@click.option('--start-date', type=str, help="The start date of the data wish to obtain in YYYY-MM-DD")
@click.option('--end-date', type=str, help="The end date of the data wish to obtain in YYYY-MM-DD")
@click.option('--write-to', type=str, help="Path to directory where raw data will be written to")

def main(url, start_date, end_date, write_to):
    """
    Command-line interface for obtaining Vancouver data within a specified date range and writing it to a specified directory.

    Parameters
    ----------
    url : str
        A string url that serves as the API endpoint to get the data from
    start_date : str
        A string in YYYY-MM-DD format (e.g. "1990-01-01") that the weather API will start extracting from.
    end_date : str
        A string in YYYY-MM-DD format (e.g. "2000-01-01") that the weather API will conclude the query.
    write_to : str
        A string path for the csv file to be stored.

    Examples
    --------
    $ python download_data.py \
        --url="https://archive-api.open-meteo.com/v1/archive" \
        --start-date=2021-01-01 \
        --end-date=2022-01-01 \
        --write-to="../data"
    
    """

    get_vancouver_data(url, start_date, end_date, write_to, create_csv = True)

if __name__ == '__main__':
    main()