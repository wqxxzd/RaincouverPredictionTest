# eda_precipitation.py
# author: Group 9
# date: 2023-11-30

import click
import os
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns

@click.command()
@click.option('--data-file', type=str, help="Path to the dataset")
@click.option('--plot-to', type=str, help="Path to directory where the plots will be written to")

def main(data_file, plot_to):
    '''Performs EDA on precipitation data and saves histograms and 
    the correlation table as PNG files.'''

    # Read the data and preprocess
    precipit_df = pd.read_csv(data_file).drop(columns = ['sunrise', 
                                                         'sunset', 
                                                         'weather_code', 
                                                         'rain_sum', 
                                                         'snowfall_sum',
                                                         'precipitation_hours', 
                                                         'date'])

    # Generate histograms for numeric columns
    alt.data_transformers.disable_max_rows()
    
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
    numeric_cols_hists.save(os.path.join(plot_to, "histogram_numeric_features.png"),
              scale_factor=2.0)

    # Generate and save correlation table
    correlation_table = precipit_df[numeric_cols].corr(method='spearman')
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_table, annot=True, cmap="gist_yarg", fmt=".2f")
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("Correlation Table")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_to, "correlation_heatmap.png"))

    
if __name__ == '__main__':
    main()
