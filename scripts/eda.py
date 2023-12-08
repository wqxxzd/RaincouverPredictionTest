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
    """
    Performs exploratory data analysis (EDA) on precipitation data. 
    This includes generating histograms for numeric features and a correlation heatmap.

    This script reads the precipitation dataset, preprocesses it by dropping certain columns, 
    and then produces histograms for all numeric columns and a correlation heatmap. 
    Both the histograms and heatmap are saved as PNG files.

    Parameters
    ----------
    data_file : str
        Path to the CSV file containing the dataset.
    plot_to : str
        Path to the directory where the plots will be saved.

    Returns
    -------
    None
        This function does not return any value but writes the generated plots to the specified path.

    Notes
    -----
    - Histograms are created for each numeric column in the dataset.
    - A correlation heatmap is generated using Spearman's rank correlation.
    - The function checks for the existence of the target directory and creates it if it does not exist.

    Examples
    --------
    Command line usage:
    $ python scripts/eda.py  \
          --data-file=data/van_weather_1990-01-01_2023-11-06.csv \
          --plot-to=results/figures
    """

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
    # write_to path transforming
    if plot_to != '':
        plot_to = plot_to if plot_to[-1] == '/' else plot_to + '/'
            
    # Check write_to path existence
    if not os.path.exists(plot_to):
        os.makedirs(plot_to)

    # Save histograms as PNG
    numeric_cols_hists.save(os.path.join(plot_to, "histogram_numeric_features.png"),
              scale_factor=2.0)

    # Generate and save correlation table
    correlation_table = precipit_df[numeric_cols].corr(method='spearman')
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_table, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title("Correlation Table", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_to, "correlation_heatmap.png"), bbox_inches='tight', dpi=300)

    
if __name__ == '__main__':
    main()
