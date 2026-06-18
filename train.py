"""
Train the SMS spam classifier and save one deployable artifact.

Model choice (see notebook for the full comparison):
  LogisticRegression(C=3), word+char hybrid TF-IDF features, NO class_weight
  balancing. An earlier version used class_weight='balanced', which traded
  too much precision for recall: for a spam filter, a false positive (a real
  message marked as spam) is costlier than a false negative (a spam message
  that slips through), so model selection used an F0.5 score (precision
  weighted 2x recall) across a small grid of models/features via 4-fold CV.
  Plain class_weight=None won by a clear margin -- see README for numbers.

Usage:
    python train.py

Produces:
    pipeline.joblib  - the fitted sklearn Pipeline (cleaning + features + model),
                       refit on the FULL dataset for deployment
    metrics.json     - honest metrics from a held-out 80/20 split, for the
                       README and the Streamlit app's "model performance" panel
"""

import json

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from preprocessing import build_pipeline, load_data

DATA_PATH = "spam.csv"
RANDOM_STATE = 2  # fixed split seed -> identical metrics on every run


def make_classifier():
    return LogisticRegression(C=3, max_iter=1000, random_state=42)


def main():
    df = load_data(DATA_PATH)
    print(f"Loaded {len(df)} messages ({df['target'].mean()*100:.1f}% spam) after dropping duplicates.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["target"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["target"],
    )

    # --- Honest evaluation on a held-out split ---
    eval_pipeline = build_pipeline(make_classifier())
    eval_pipeline.fit(X_train, y_train)
    y_pred = eval_pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "n_test": int(len(y_test)),
        "n_train": int(len(X_train)),
    }

    print(f"\nAccuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1       : {metrics['f1']:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # --- Refit on the FULL dataset for the artifact that actually gets deployed ---
    # (the split above exists only to report trustworthy numbers; once that's
    # done, training on every available row gives the shipped model more
    # signal to work with)
    final_pipeline = build_pipeline(make_classifier())
    final_pipeline.fit(df["text"], df["target"])
    joblib.dump(final_pipeline, "pipeline.joblib")

    print("Saved pipeline.joblib (trained on the full dataset) and metrics.json")
    print("Run: streamlit run app.py")


if __name__ == "__main__":
    main()