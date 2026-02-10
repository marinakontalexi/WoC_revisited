# Reproducing the Plots

This repository contains the code used to reproduce the figures in the paper.

## Dependencies

* Python 3.x
* numpy
* pandas
* scipy
* matplotlib
* seaborn
* jupyter

## Execution Order

### 1. Data preprocessing

```bash
python data_processing.py
```

Loads and merges raw experimental data and produces an intermediate dataset.

### 2. Data cleaning

```bash
python data_cleaning.py
```

Filters invalid responses, removes outliers, and produces the final datasets used in the analysis.

### 3. Plot generation

```bash
jupyter notebook plots.ipynb
```

Running all cells reproduces the plots in the paper.

## Code Structure

* `data_processing.py`: merges raw data and computes basic variables
* `data_cleaning.py`: filters and cleans the dataset
* `dataset_functions.py`: task-level statistics and aggregation methods
* `general_utils.py`: shared helper functions
* `plots.ipynb`: generates all figures
