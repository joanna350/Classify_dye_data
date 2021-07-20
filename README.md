### Project Scope
- Run classifiers on a real world data from a dye factory
- XGBoost performs optimally, shap analysis to be done

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
* `chmod u+x` first if sudo error occurs

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
* Returns `test.template.csv` on the same directory only when `others` == 0 (based on xgbclassifier)

* Returns `finetuned.booster` on the same directory with `gridsearch` == 1
* Returns `test.template.csv` under a directory with the current timestamp with `gridsearch` == 1 if `finetuned.booster` is already saved

### Directory structure 

* Run in the sequence of returns. Optional `plot_cnf` == 1 at the last step

```
|-- Model
|   |-- run.sh
|   |-- README.md
|   |-- data
|   |   |-- train.csv
|   |   |-- train.out.csv
|   |   |-- test.csv
|   |-- Field Description.xlsx
|   |-- utility.py
|   |-- classifier.py
|   |-- test.template.csv
|   |-- finetuned.booster
|   |-- plots
|   |   |--xgb_plot_importance.png
|   |   |--xgbclassifier_pr_curve.png
|   |   |--xgbclassifier_ROC_curve.png
|   |   |--finetuned.booster_plot_importance.png
|   |   |--finetuned.booster_pr_curve.png
|   |   |--finetuned.booster_ROC_curve.png
|   |   |--finetuned.booster_confusion_matrix.png
|   |-- 2021-02-28 01/54/42
|   |-- |-- test.template.csv
```
