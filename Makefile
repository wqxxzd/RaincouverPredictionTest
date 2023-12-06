# Define variables for directories and files to make the Makefile more maintainable
PROCESSED_DATA_DIR = data/processed
FIGURES_DIR = results/figures
MODELS_DIR = results/models
REPORTS_DIR = reports/milestone4

# Run all the datafiles
all : download_data eda drop_split_preprocess classification build_report

download_data : scripts/download_data.py 
	python scripts/download_data.py \
        --url="https://archive-api.open-meteo.com/v1/archive" \
        --start-date=1990-01-01 \
        --end-date=2023-11-06\
        --write-to="./data"

eda : scripts/eda.py \
	data/van_weather_1990-01-01_2023-11-06.csv
	python scripts/eda.py \
  		--data-file=data/van_weather_1990-01-01_2023-11-06.csv \
  		--plot-to=$(FIGURES_DIR)

drop_split_preprocess : scripts/drop_split_preprocess.py \
	data/van_weather_1990-01-01_2023-11-06.csv
	python scripts/drop_split_preprocess.py \
		--data-file=data/van_weather_1990-01-01_2023-11-06.csv \
		--data-to=$(PROCESSED_DATA_DIR)  \
		--preprocessor-to=$(MODELS_DIR) \
		--seed=522

classification : scripts/classification.py \
	$(PROCESSED_DATA_DIR)/X_train.csv \
	$(PROCESSED_DATA_DIR)/X_test.csv \
	$(PROCESSED_DATA_DIR)/X_train.csv \
	$(PROCESSED_DATA_DIR)/X_train.csv \
	$(MODELS_DIR)/precipit_preprocessor.pickle
	python scripts/classification.py\
		--x_train=$(PROCESSED_DATA_DIR)/X_train.csv \
		--y_train=$(PROCESSED_DATA_DIR)/y_train.csv \
		--x-test=$(PROCESSED_DATA_DIR)/X_test.csv \
		--y-test=$(PROCESSED_DATA_DIR)/y_test.csv \
		--preprocessor=$(MODELS_DIR)/precipit_preprocessor.pickle \
		--columns-to-drop=parameter/columns_to_drop.csv \
		--pipeline-to=$(MODELS_DIR) \
		--plot-to=$(FIGURES_DIR) \
		--seed=522

build_report : $(REPORTS_DIR)/raincouver_prediction_report4.ipynb \
    $(FIGURES_DIR)/classification_report.png \
	$(FIGURES_DIR)/correlation_heatmap.png \
	$(FIGURES_DIR)/Feature_importance.png \
	$(FIGURES_DIR)/histogram_numeric_features.png \
	$(FIGURES_DIR)/model_comparison.png 
	jupyter-book build $(REPORTS_DIR)

# Clean up the generated files
clean:
	rm -rf data
	rm -rf $(FIGURES_DIR)
	rm -rf $(MODELS_DIR)
	rm -rf $(REPORTS_DIR)/_build