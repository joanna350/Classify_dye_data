# XGBoost

### Environment
```
MacOS Catalina v10.15.7
Python 3.9.1
```

### Dependency
```
numpy==1.19.5
sklearn==0.24.0
xgboost==1.3.1
pandas==1.2.0
matplotlib==3.3.3
seaborn==0.11.1
imblearn==0.7.0
argparse==1.1
logging==0.5.1.2
```

### Installation
```
python(version) -m pip(v) install --user -Iv '__package_name__==__version__'
```

## To Run

### Command Line Interface

* `./run.sh` prepares the directory structure for operation

* To test the models:
```
# For description
python_v classifier.py -h

# For operation
python_v classifier.py arg1 arg2 arg3'
```
* `plot_cnf`: arg1 (default 0) determines the plot of confusion matrix
* `gridsearch`: arg2 (default 0) for hyperparameter tuning
* `others`: arg3 (default 0) to run other models
* Returns `test.template.csv` only when `others` == 0

### Directory structure (e.g. run default)

```
|-- Model
|   |-- run.sh
|   |-- Readme.md
|   |-- data
|   |   |-- train.csv
|   |   |-- train.out.csv
|   |   |-- test.csv
|   |-- File Description.xlsx
|   |-- utility.py
|   |-- classifier.py
|   |-- test.template.csv
|   |-- plots
|   |   |--xgb_plot_importance.png
|   |   |--xgbclassifier_pr_curve.png
|   |   `--xgbclassifier_ROC_curve.png
```
