import click
import os
import altair as alt
import numpy as np
import pandas as pd
import pickle
from sklearn import set_config
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import fbeta_score, make_scorer
from joblib import dump
import sys
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn import set_config

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.utils import cross_val_model

@click.command()
@click.option('--x_train', type=str, help="Path to X training data")
@click.option('--y_train', type=str, help="Path to y training data")
@click.option('--x-test', type=str, help="Path to x test data")
@click.option('--y-test', type=str, help="Path to y test data")
@click.option('--preprocessor', type=str, help="Path to preprocessor object", default=None)
@click.option('--columns-to-drop', type=str, help="Optional: columns to drop")
@click.option('--pipeline-to', type=str, help="Path to directory where the pipeline object will be written to")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)

def main(x_train, y_train, x_test, y_test, preprocessor, columns_to_drop,
         pipeline_to, plot_to, seed): 
    """
    Trains classifiers on the provided dataset, evaluates their performance, 
    and saves the best performing model and relevant plots.

    This function reads training and test datasets, optionally applies preprocessing,
    compares multiple classifiers, and performs hyperparameter optimization on the best model. 
    It saves the optimized pipeline and generates plots showing feature importance and model comparison.

    Parameters
    ----------
    x_train : str
        Path to the CSV file containing X training data.
    y_train : str
        Path to the CSV file containing y training data.
    x_test : str
        Path to the CSV file containing X test data.
    y_test : str
        Path to the CSV file containing y test data.
    preprocessor : str
        Path to the preprocessor object. Default is None.
    columns_to_drop : str
        Optional: Path to CSV file containing names of columns to drop.
    pipeline_to : str
        Path to the directory where the pipeline object will be saved.
    plot_to : str
        Path to the directory where plots will be saved.
    seed : int
        Random seed for reproducibility. Default is 123.

    Returns
    -------
    None
        This function does not return any value but writes the optimized pipeline object 
        and plots to the specified paths.

    Notes
    -----
    - The function uses a variety of classifiers including Decision Tree, KNN, SVM, and Logistic Regression.
    - Hyperparameter optimization is performed on the best model based on the F1 score.
    - The function generates plots for feature importance, model comparison, and a classification report.
    - The function will exit if the best model is not SVM RBF, as it is set up to optimize this specific model.

    Examples
    --------
    Command line usage:
    $ python scripts/classification.py \
             --x_train=data/processed/X_train.csv \
             --y_train=data/processed/y_train.csv \
             --x-test=data/processed/X_test.csv \
             --y-test=data/processed/y_test.csv \
             --preprocessor=results/models/precipit_preprocessor.pickle \
             --columns-to-drop=data/processed/columns_to_drop.csv \
             --pipeline-to=results/models \
             --plot-to=results/figures \
             --seed=522
    """

    np.random.seed(seed)
    set_config(transform_output="pandas")

    #Read in data and preprocessor
    X_train = pd.read_csv(x_train)
    y_train = pd.read_csv(y_train)
    
    y_col_name = y_train.columns.tolist()[0]
    y_train_class = y_train[y_col_name]
    preprocess = pickle.load(open(preprocessor, "rb"))
    
    if columns_to_drop:
        to_drop = pd.read_csv(columns_to_drop).feats_to_drop.tolist()
        X_train = X_train.drop(columns=to_drop)

    click.echo(f'Running model training for columns {X_train.columns.tolist()}')

    #Model parameter to be explored
    models = {
    "Decision Tree": DecisionTreeClassifier(random_state=522),
    "KNN": KNeighborsClassifier(),
    "RBF SVM": SVC(random_state=522),
    "Logistic Regression": LogisticRegression(max_iter=2000, multi_class="ovr", random_state=522),
    }

    #Plot out feature importance
    pipe = make_pipeline(preprocess, models["Logistic Regression"])
    pipe.fit(X_train, y_train_class)
    numeric_features = pipe.named_steps['columntransformer'].named_transformers_['standardscaler'].get_feature_names_out().tolist()
    coefficients = pipe.named_steps['logisticregression'].coef_[0]
    
    feature_importance = pd.DataFrame({'Feature': numeric_features, 'Importance': np.abs(coefficients)})
    feature_importance = feature_importance.sort_values('Importance', ascending=True)   
    plt.figure(figsize=(12, 10))
    feature_importance.plot(x='Feature', y='Importance', kind='barh')
    plt.title("Feature_importance")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_to, "Feature_importance.png"))

    click.echo(f'Feature importance has been parked at {os.path.join(plot_to, "Feature_importance.png")}')

    #Model type selection
    classification_metrics = ["accuracy", "precision", "recall", "f1"]
    results_df = cross_val_model(preprocess, models, X_train, y_train_class, classification_metrics)

    #Choose the model based on the best F1 score. 
    result_dict = results_df.loc['test_f1', :].to_dict()

    # Create a bar chart with F1 scores all the tested models and publish it
    plt.figure(figsize=(8, 8))
    plt.bar(models.keys(), result_dict.values(), color='blue')
    plt.xlabel('Models')
    plt.ylabel('Test F1 Score')
    plt.title('Test F1 Scores for Different Models')
    plt.ylim(0.8, 0.9)  # Set the y-axis limit between 0 and 1 for F1 scores
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.tight_layout()
    plt.savefig(os.path.join(plot_to, "model_comparison.png"))
    
    click.echo(f'Model F1 performance has been parked at {os.path.join(plot_to, "model_comparison.png")}')
    # Save the bar chart as an image
    plt.tight_layout()
    
    #Get model name with max_score
    max_score = max(result_dict, key=result_dict.get)
    model_name = max_score[0]

    #Echo out if the model chosen is not expected as SVM RBF
    if model_name == 'RBF SVM':
        click.echo('Proceed with hyperparameter optimization for RBF SVM model')
    else:
        click.echo('Best model is not expected SVM RBF. Other model parameters required for optimization, exitting the program')
        sys.exit(0)

    #Hyperparameter optimization
    param_grid = {"svc__C": 10.0**np.arange(-3,3)}
    svc_pipe = make_pipeline(preprocess, models[model_name])
    grid_search = GridSearchCV(svc_pipe,param_grid=param_grid,n_jobs=-1,return_train_score=True)
    grid_search.fit(X_train, y_train_class)
    C_best_value = grid_search.best_params_['svc__C']
    opt_pipe = make_pipeline(preprocess, SVC(C=C_best_value, random_state = 522))
    
    opt_pipe.fit(X_train, y_train_class)

    #Saving optimum pipe as a pickle file for testing
    with open(os.path.join(pipeline_to, "optimum_cls_svm_pipeline.pickle"), 'wb') as f:
        pickle.dump(opt_pipe, f)

    click.echo(f'Optimized model has been parked at {os.path.join(pipeline_to, "optimum_cls_svm_pipeline.pickle")}')

    #Saving model evaluation result
    X_test = pd.read_csv(x_test)
    y_test = pd.read_csv(y_test)
    y_col_name = y_test.columns.tolist()[0]
    y_test_class = y_test[y_col_name]

    if columns_to_drop:
        to_drop = pd.read_csv(columns_to_drop).feats_to_drop.tolist()
        X_test = X_test.drop(columns=to_drop)
        
    click.echo(f'Running model test for columns {X_test.columns.tolist()}')        
    y_pred = opt_pipe.predict(X_test)

    report_dict = classification_report(y_test_class, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    
    report_df = report_df.loc[['False', 'True']].round(2)
    report_df.index = ['No rain', 'Rain']
    
    plt.figure(figsize=(5, 1))  # Adjusting the figure size
    ax = plt.gca()
    table = ax.table(cellText=report_df.values,
                     colLabels=report_df.columns,
                     rowLabels=report_df.index,
                     loc='center',
                     cellLoc='center')
    table.scale(1, 1.5) 
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_to, "classification_report.png"), dpi=300, bbox_inches='tight')
    click.echo(f'Classification report has been parked at {os.path.join(plot_to, "classification_report.png")}')

if __name__ == '__main__':
    main()
