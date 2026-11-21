# these are all teh imports we are gonna use in this task
#this is a genral use artifical neural network meaning it will
#work on all datasets that are tabular

import argparse
import json
import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import(
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler 
from sklearn.utils.class_weight import compute_class_weight
import keras

risk_bins = [-0.001, 0.33, 0.66, 1.001]
risk_lables = ["Low", "Medium", "High"]
positve_gueesses = ["yes", "1", "true", "churn", "positive", "pass", "y"]

def section(title):
    print("")
    print("="*70)
    print(title)
    print("="*70)

def save_file(fig, path, show):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"[saved] {path}")
    if show == True:
        plt.show()
    plt.close(fig)

def parse_args():
    p = argparse.ArgumentParser(
        description = "Genral purpose ANN classifier" \
        "using tensorflow/keras",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--data", default="Telco-Customer-Churn.csv", 
                   help="path to csv file."
                   )
    p.add_argument("--target", default="Churn", 
                   help="name of target column.")
    p.add_argument("--positive-label", default=None,
                   help= "binary targets only: value treated as class 1 autodetect if ommited")
    p.add_argument("--drop-cols", nargs="*", default=[],
                   help="columns to ignore (like ids, names and otehrs) unique-ID etxt columns" \
                   "dropped automatically.")
    p.add_argument("--test-size", type=float, default=0.3,
                   help="fractions used for testing part of the ANN")
    p.add_argument("--hidden-layers", type=int, nargs="+", default=[16, 8],
                   help="neurons in each layer like hidden layers 64 32 16")
    p.add_argument("--dropout", type=float, default=0.2,
                   help=" the dropout rate (0 means disabled)")
    p.add_argument("--epochs", type=int, default=100, 
                   help="maximum training epochs")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=10,
                   help="early-stoping patience/epochs.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="binary decsion threshold")
    p.add_argument("--balance-classes", action="store_true",
                   help="weight classes inversly to their frequency(helps imbalanced data)")
    p.add_argument("--predict-file", "--predicted-file", dest="predict_file",
                   default=None,
                   help="optimal csv of new records(same columns no target)" \
                   "to score after training")
    p.add_argument("--output-dir", default="outputs",
                   help="folde for plots, model and predictions")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-show", action="store_true", 
                   help="save plot without opening windows.")
    return p.parse_args()

#now we will load data, clean it and analyse it

def load_data(path):
    if not os.path.exists(path):
        sys.exit(f"Error: file not found: {path} \n"
                 f"pass the correct location with --data \"path/to/file.csv\"")
    return pd.read_csv(path)

def clean_dataframe(df: pd.DataFrame, target=None):
    """strip whitespaces, turn blank strings into NaN, convert numeric-looking text"""
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            s = df[column].astype("string").str.strip()
            s = s.replace("", pd.NA)
            numeric = pd.to_numeric(s, errors="coerce")
            if numeric.notna().sum() == s.notna().sum():
                df[column] = numeric
            else:
                df[column] = s
    if target is not None and target in df.columns:
        df = df.dropna(subset=[target])

    for column in df.columns:
        if not pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].astype(object).where(df[column].notna(), np.nan)
    return df

def analyse_data_finaly(df: pd.DataFrame, target, out_dir, show):
    print("shape (rows, columns): ", df.shape)
    print("")
    print("first 5 rows: ")
    print(df.head())
    print("")
    print("column Info: ")
    df.info()
    print("")
    print("statistical summary (numeric): ")
    print(df.describe())
    print("")
    print("missing value per column: ")
    print(df.isnull().sum()[df.isnull().sum()>0]
          if df.isnull().any().any() else " none")
    print("")
    print("duplicate rows: ", df.duplicated().sum())
    countering = df[target].value_counts()
    print("")
    print(f"target '{target}' distribution: ")
    print(countering)
    print((countering/countering.sum()*100).round(2).astype(str) + "%")

#target distribution

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=df[target].astype(str), order=countering.index.astype(str),
                  ax=ax)
    ax.set_title(f"target distribution: {target}")
    save_file(fig, os.path.join(out_dir, "01_target_distribution.png"), show)

#numeric feature hsitograms
    num_of_cols = [columns for columns in 
                   df.select_dtypes(include="number").columns
                   if columns != target][:12]
    if num_of_cols:
        n_columns = min(3, len(num_of_cols))
        n_rows = int(np.ceil(len(num_of_cols)/ n_columns))
        fig, axes = plt.subplots(n_rows, n_columns, squeeze=False,
                                 figsize=(5*n_columns, 3.5*n_rows))
        for ax,c in zip(axes.ravel(), num_of_cols):
            sns.histplot(df[c].dropna(), kde=True, ax=ax)
            ax.set_title(c)
        for ax in axes.ravel()[len(num_of_cols):]:
            ax.axis("off")
        save_file(fig, os.path.join(out_dir,"02_numeric_distributions.png"),
                 show)

    #correlation heatmap

    if len(num_of_cols):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(df[num_of_cols].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", ax=ax)
        ax.set_title("correlation of numeric features")
        save_file(fig, os.path.join(out_dir, "03_correlation_heatmap.png"),
                  show)

#pre processing 

def select_features(df: pd.DataFrame, target, drop_cols):
    #remove target, user chosen columns and ID-like columns and constant columns

    features = df.drop(columns=[target])
    user_dropper = [column for column in drop_cols if column in features.columns]


    id_columns = [
        column for column in features.columns
        if column not in user_dropper
        and not pd.api.types.is_numeric_dtype(features[column])
        and features[column].nunique(dropna=True) == len(features)
    ]

    conts_cols = [column for column in features.columns
                  if column not in user_dropper 
                  and features[column].nunique(dropna=True)<=1]

    to_drop = list(dict.fromkeys(user_dropper+id_columns+conts_cols))
    ids = df[(user_dropper + id_columns)[0]] if (user_dropper+id_columns) else None 
    return features.drop(columns=to_drop), ids, to_drop

def encode_target(y_rawm: pd.DataFrame, positive_label = None):
    #return integer encoded targets and teh list of class names (index=code)
    y_rawm = y_rawm.copy()
    if pd.api.types.is_float_dtype(y_rawm) and (y_rawm.dropna() % 1 == 0).all():
        y_rawm = y_rawm.astype(int)
    y_stringer = y_rawm.astype(str)
    classes = sorted(y_stringer.unique())

    if len(classes)<2:
        sys.exit("ERROR: target the column has only one class")

    if len(classes) == 2:
        if positive_label is not None:
            if str(positive_label) not in classes:
                sys.exit(f"ERROR: --positive-label '{positive_label}' not in {classes}")
            pos = str(positive_label)
        else:
            pos = next((column for column in classes if column.lower()in positve_gueesses),
                       classes[1])
        neg = next(column for column in classes if column != pos)
        codes = (y_stringer == pos).astype(int)
        return codes, [neg, pos]

    le = LabelEncoder()
    codes = pd.Series(le.fit_transform(y_stringer), index=y_stringer.index)
    return codes, list(le.classes_)

def build_preprocessor(x):
    numeric_cols= [column for column in x.columns if pd.api.types.is_numeric_dtype(x[column])]
    categorical_cols= [column for column in x.columns if column not in numeric_cols]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ])
    return preprocessor, numeric_cols, categorical_cols

#ANN model

def build_model(input_dim, n_classes, hidden_layers, dropout, learning_rate):
    model = keras.Sequential(name="ann_clasifier")
    model.add(keras.layers.Input(shape=(input_dim,)))
    for units in hidden_layers:
        model.add(keras.layers.Dense(units, activation="relu"))
        if dropout>0:
            model.add(keras.layers.Dropout(dropout))

    if n_classes == 2:
        model.add(keras.layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
    else:
        model.add(keras.layers.Dense(n_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    return model

def predict_classes(model, x, n_classes, threshold):
    #return (labels, probabilities). binary: P(class).
    #multi-class: full matrix
    proba = model.predict(x, verbose=0)
    if n_classes == 2:
        proba = proba.ravel()
        labels = (proba >= threshold).astype(int)
    else:
        labels = np.argmax(proba, axis=1)
    return labels, proba

def risk_level(probabilities):
    return pd.cut(probabilities, bins=risk_bins, labels=risk_lables)

#evaluation

def plot_history(history, out_dir, show):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="validation")
    axes[0].set_title("loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="validation")
    axes[1].set_title("accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    save_file(fig, os.path.join(out_dir, "04_training_curves.png"), show)

def evaluate_model(y_test, labels, proba, class_names, out_dir, show):
    acc = accuracy_score(y_test, labels)
    cm = confusion_matrix(y_test, labels)
    metrics = {"accuracy": float(acc)}

    print(f"accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print("")
    print("codusion matix (row = actual, columns = predictions): ")
    print(cm)
    print("")
    print("classification report: ")
    print(classification_report(y_test, labels, target_names= class_names,
                                 zero_division=0))

    if len(class_names) == 2:
        auc = roc_auc_score(y_test, proba)
        metrics["roc_auc"] = float(auc)
        print(f"Roc-Auc: {auc:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title("confusion matrix")
    save_file(fig, os.path.join(out_dir, "05_confusion_matrix.png"), show)
    return metrics

#predicting new records

def score_new_file(path, model, preprocessor, feature_cols, numeric_cols,
                   class_names, threshold, out_dir):
    new_df = clean_dataframe(load_data(path))
    missing = [cols for cols in feature_cols if cols not in  new_df.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing required columns: {missing}")

    x_new = new_df[feature_cols].copy()
    for cols in numeric_cols:
        x_new[cols] =  pd.to_numeric(x_new[cols], errors="coerce")

    labels, proba = predict_classes(model, preprocessor.transform(x_new),
                                    len(class_names), threshold)

    result = new_df.copy()
    result["predicted"] = [class_names[i] for i in labels]
    if len(class_names) == 2:
        result[f"probability_{class_names[1]}"] = np.round(proba, 4)
        result["risk_level"] = risk_level(proba)
    else:
        result["confidence"]= np.round(proba.max(axis=1), 4)

    out_path = os.path.join(out_dir, "new_predictions.csv")
    result.to_csv(out_path, index=False)
    print(f"scored {len(result)} new records -> {out_path}")
    print(result.head(10).to_string(index=False))

#now we move onto the main workflow of this thing

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    keras.utils.set_random_seed(args.seed)
    show = not args.no_show

    section("1. data analysis")
    df = clean_dataframe(load_data(args.data), args.target)
    if args.target not in df.columns:
        sys.exit(f"ERROR: target column '{args.target}' not found"
                 f"available columns: {list(df.columns)}")
    analyse_data_finaly(df, args.target, args.output_dir, show)


    section("2. pre-processing")
    x, ids, dropped = select_features(df, args.target, args.drop_cols)
    y, class_names = encode_target(df[args.target], args.positive_label)
    n_classes = len(class_names)
    task = "binary" if n_classes == 2 else f"multi-class ({n_classes} classes)"
    print(f"task types      : {task}")
    print(f"class encoding  : {dict(enumerate(class_names))}")
    print(f"dropped columns : {dropped if dropped else 'none'}")
    print(f"feature columns : {list(x.columns)}")


    #train test split
    section(f"3. train test split ({round((1- args.test_size)* 100)}/"
            f"{round(args.test_size * 100)})")
    startify = y if y.value_counts().min() >=2 else None
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size,
        random_state=args.seed, stratify=startify
    )
    print(f"training sample: {len(x_train)} testing sample : {len(x_test)}")

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(x_train)
    x_train_p = preprocessor.fit_transform(x_train)
    x_test_p = preprocessor.transform(x_test)
    print(f"numeric columns ({len(numeric_cols)}): scaled with Standard scaler")
    print(f"categorical columns ({len(categorical_cols)}): one-hot encoded")
    print(f"input size after encoding: {x_train_p.shape[1]} features")

    section("4. ANN modle tendorflow/keras")
    model = build_model(x_train_p.shape[1], n_classes, args.hidden_layers,
                        args.dropout, args.learning_rate)
    model.summary()

    class_weight = None
    if args.balance_classes:
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight = {int(c): float(w) for c,w in zip(classes, weights)}
        print("class weights:", class_weight)

    history = model.fit(
        x_train_p, y_train.values,
        validation_split = 0.2,
        epochs = args.epochs,
        batch_size= args.batch_size,
        callbacks= [keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True)],
        class_weight = class_weight,
        verbose=2)
    
    print(f"training stopped after {len(history.history['loss'])} epochs")
    plot_history(history, args.output_dir, show)

    section("5. evaluation on test")
    labels, proba = predict_classes(model, x_test_p, n_classes, args.threshold)
    metrics = evaluate_model(y_test, labels, proba, class_names, args.output_dir, show)

    section("6. prediction (risk)")
    preds = pd.DataFrame(index= x_test.index)
    if ids is not None:
        preds[ids.name] = ids.loc[x_test.index]
    preds["actual"] = [class_names[i] for i in  y_test.values]
    preds["predicted"] = [class_names[i] for i in  labels]
    if n_classes == 2:
        prob_col = f"probability_{class_names[1]}"
        preds[prob_col] = np.round(proba, 4)
        preds["risk_level"] = risk_level(proba)
        print(f"risk level: low < {risk_bins[1]}, medium < {risk_bins[2]} high above")
        print("risk level counts (test sets): ")
        print(preds["risk_level"].value_counts().sort_index())
        print("")
        print(f"10 highest rank records ('{class_names[1]}'): \n",
        preds.sort_values(prob_col, ascending=False).head(10).to_string(index=False))
    else:
        preds["confidence"] = np.round(proba.max(axis=1), 4)
        print(preds.head(10).to_string(index=False))

    preds_path = os.path.join(args.output_dir, "test_predictions.csv")
    preds.to_csv(preds_path, index= False)
    print("")
    print(f"[saved] {preds_path}")

    model.save(os.path.join(args.output_dir, "ann_model.keras"))
    joblib.dump({"preprocessor": preprocessor, "feature_col": list(x.columns),
                 "numeric_columns": numeric_cols, "class_names": class_names},
                 os.path.join(args.output_dir, "preprocessor.joblib"))
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  [saved] model, preprocessor and metrics in '{args.output_dir}/'")

    if args.predict_file:
        section("7. scoring new records")
        score_new_file(args.predict_file, model, preprocessor,
                       list(x.columns), numeric_cols, class_names,
                       args.threshold, args.output_dir)

if __name__ == "__main__":
    main()
