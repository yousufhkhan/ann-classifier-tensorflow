# General-Purpose ANN Classifier (TensorFlow / Keras)

A single-script, command-line Artificial Neural Network for **any tabular CSV
dataset** with a binary or multi-class label. It was built for a university AI
lab on telecom customer churn, but nothing in it is specific to that dataset:
you choose the CSV, the target column and the network shape from the command line.

## What it does

| Step | What happens |
|------|--------------|
| 1. Data analysis | Shape, info, statistics, missing values, duplicates, class balance, plus saved plots (target distribution, numeric histograms, correlation heatmap) |
| 2. Pre-processing | Trims whitespace, fixes numbers stored as text, drops ID/constant columns, fills missing values, one-hot encodes categories, scales numbers |
| 3. Split | Stratified train/test split (default 70/30). Scalers and encoders are fitted on training data only, so there is no data leakage |
| 4. ANN | Keras `Sequential` model with configurable hidden layers, dropout, Adam optimizer and early stopping |
| 5. Evaluation | Accuracy, confusion matrix, precision/recall/F1 report, ROC-AUC (binary), training curves |
| 6. Prediction | Probability and Low / Medium / High risk level for each test record. Optionally scores a new CSV |

Binary targets use a sigmoid output with binary cross-entropy. Targets with
three or more classes automatically switch to softmax with categorical cross-entropy.

## Setup

Use a virtual environment so TensorFlow's dependencies don't clash with your other projects.

**Windows**
```
python -m venv venv_ann
venv_ann\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**
```
python3 -m venv venv_ann
source venv_ann/bin/activate
pip install -r requirements.txt
```

> TensorFlow supports only some Python versions. If `pip` cannot find
> `tensorflow`, create the environment with Python 3.10, 3.11 or 3.12.

## Dataset

Datasets are not stored in this repository. For the churn example, download
**Telco Customer Churn** from Kaggle
(https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place the
CSV next to `ANN_Classifier.py`.

## Usage

Telco churn (the defaults):
```
python ANN_Classifier.py --data Telco-Customer-Churn.csv --target Churn
```

Any other dataset:
```
python ANN_Classifier.py --data my_data.csv --target label --drop-cols id name
```

Bigger network, imbalanced data, no pop-up plot windows:
```
python ANN_Classifier.py --data Telco-Customer-Churn.csv --target Churn \
    --hidden-layers 64 32 16 --balance-classes --no-show
```

Train, then score new customers who have no label yet:
```
python ANN_Classifier.py --data Telco-Customer-Churn.csv --target Churn --predict-file new_customers.csv
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data` | `Telco-Customer-Churn.csv` | Path to the CSV file |
| `--target` | `Churn` | Name of the label column |
| `--positive-label` | auto | Binary only: the value to treat as class 1 (e.g. `Yes`) |
| `--drop-cols` | none | Columns to ignore. Unique-ID text columns are dropped automatically |
| `--test-size` | `0.30` | Fraction of data used for testing |
| `--hidden-layers` | `16 8` | Neurons per hidden layer |
| `--dropout` | `0.2` | Dropout rate (`0` disables it) |
| `--epochs` | `100` | Maximum epochs (early stopping usually ends sooner) |
| `--batch-size` | `32` | Batch size |
| `--learning-rate` | `0.001` | Adam learning rate |
| `--patience` | `10` | Early-stopping patience |
| `--threshold` | `0.5` | Binary decision threshold |
| `--balance-classes` | off | Weight classes by inverse frequency |
| `--predict-file` | none | CSV of new records to score after training |
| `--output-dir` | `outputs` | Where results are saved |
| `--seed` | `42` | Random seed for reproducibility |
| `--no-show` | off | Save plots without opening windows |

## Output

Everything is written to `outputs/` (git-ignored):

- `01_target_distribution.png`, `02_numeric_distributions.png`, `03_correlation_heatmap.png`
- `04_training_curves.png`, `05_confusion_matrix.png`
- `test_predictions.csv`: actual vs predicted label, probability and risk level
- `new_predictions.csv`: only when `--predict-file` is used
- `ann_model.keras`, `preprocessor.joblib`, `metrics.json`: the trained model and what is needed to reuse it

## Results

_Fill in after running, for example:_

| Metric | Value |
|--------|-------|
| Accuracy | ... |
| ROC-AUC | ... |

Add a screenshot of the confusion matrix by copying it into an `images/` folder and linking it here.

## Notes

- Churn data is imbalanced (about 27% of customers churn), so accuracy alone
  can be misleading. Look at recall for the churn class and the ROC-AUC, and try `--balance-classes`.
- Results vary slightly between runs and machines even with a fixed seed.
- Every class label must appear at least twice in the data so the split can be stratified.

## Tech stack

Python, TensorFlow/Keras, scikit-learn, pandas, NumPy, Matplotlib, Seaborn.