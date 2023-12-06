import click
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import set_config
import pickle

@click.command()
@click.option('--x-test', type=str, help="Path to x test data")
@click.option('--y-test', type=str, help="Path to y test data")
@click.option('--columns-to-drop', type=str, help="Optional: columns to drop", default=None)
@click.option('--pipeline-from', type=str, help="Path to directory where the fit pipeline object lives")
@click.option('--results-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)

def main(x_test, y_test, columns_to_drop, pipeline_from, results_to, seed):
    '''Evaluate the raincouver classifier and saves the evaluation results'''
    np.random.seed(seed)
    set_config(transform_output="pandas")

    #Read in data and pipeline object
    X_test = pd.read_csv(x_test)
    y_test = pd.read_csv(y_test)
    y_col_name = y_test.columns.tolist()[0]
    y_test_class = y_test[y_col_name]
    
    with open(pipeline_from, 'rb') as f:
        opt_pipe = pickle.load(f)
    
    if columns_to_drop:
        to_drop = pd.read_csv(columns_to_drop).feats_to_drop.tolist()
        print(to_drop)
        print(X_test.columns.tolist())
        X_test = X_test.drop(columns="month_cos")

    # Create the plot
    y_pred = opt_pipe.predict(X_test)

    report_dict = classification_report(y_test_class, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    report_df = report_df.loc[['False', 'True']].round(2)
    report_df.index = ['No rain', 'Rain']
    
    plt.figure(figsize=(6, 6))
    plt.table(cellText=report_df.values,
              colLabels=report_df.columns,
              rowLabels=report_df.index,
              loc='center')
    plt.axis('off')
    plt.savefig(os.path.join(plot_to, "classification_report.png"))
    click.echo(f'Classification report has been parked at {os.path.join(plot_to, "classification_report.png")}')
    
if __name__ == '__main__':
    main()
