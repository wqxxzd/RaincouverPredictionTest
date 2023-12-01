# eda_precipitation.py
# author: Group 9
# date: 2023-11-30

import click
import os
import pandas as pd
import altair as alt
import dataframe_image as dfi

@click.command()
@click.option('--data-file', type=str, help="Path to the dataset")
@click.option('--plot-to', type=str, help="Path to directory where the plots will be written to")

def main(data_file, output_path):
    '''Performs EDA on precipitation data and saves histograms and correlation table as PNG files.'''

    # Read the data and preprocess
    precipit_df = pd.read_csv(data_file).drop(columns = ['sunrise', 'sunset', 'weather_code', 'rain_sum', 'snowfall_sum', 'precipitation_hours', 'date'])

    alt.data_transformers.disable_max_rows()

    # Generate histograms for numeric columns
    numeric_cols = precipit_df.select_dtypes(include=['number']).columns.tolist()

    numeric_cols_hists = alt.Chart(precipit_df).mark_bar().encode(
        alt.X(alt.repeat(), type='quantitative', bin=alt.Bin(maxbins=40)),
        y='count()',
    ).properties(
        height=100,
        width=200
    ).repeat(
        numeric_cols,
        columns=4
    )
    # Save histograms as PNG
    numeric_cols_hists.save(f'{output_path}/numeric_cols_hists.png', scale_factor=2.0)

    # Generate and save correlation table
    correlation_table = precipit_df[numeric_cols].corr(method='spearman')
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_table, annot=True, cmap="gist_yarg")
    plt.title("Correlation Table")
    plt.savefig(f'{output_path}/correlation_table.png')

if __name__ == '__main__':
    main()
