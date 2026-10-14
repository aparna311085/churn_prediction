"""
model.py
--------
Builds, trains and evaluates a TensorFlow/Keras MLP binary classification
model for customer churn prediction.

Architecture:
  Input -> Dense(128, relu) -> BatchNorm -> Dropout(0.4)
        -> Dense(64, relu)  -> BatchNorm -> Dropout(0.3)
        -> Dense(32, relu)  -> Dropout(0.2)
        -> Dense(1, sigmoid)

Training details:
  - Adam optimizer with learning rate 1e-3
  - Binary cross-entropy loss
  - Early stopping on val_loss (patience=10)
  - ReduceLROnPlateau callback
  - Class weights to handle 73/27 imbalance
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # Suppress TF info messages

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
import joblib

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(ROOT, "data")
MODEL_DIR = os.path.join(ROOT, "outputs", "model")
CHART_DIR = os.path.join(ROOT, "outputs", "model_charts")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(MODEL_DIR, "churn_mlp_model.keras")
RESULTS_PATH = os.path.join(MODEL_DIR, "evaluation_results.json")

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_splits():
    """Load preprocessed numpy arrays from disk."""
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_val   = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_val   = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    print(f"[MODEL] Shapes -- X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_dim: int, learning_rate: float = 1e-3) -> keras.Model:
    """Build the MLP model."""
    inputs = keras.Input(shape=(input_dim,), name="features")

    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="churn_probability")(x)

    model = keras.Model(inputs, outputs, name="Churn_MLP")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy",
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")]
    )
    return model


def get_callbacks(ckpt_path: str) -> list:
    """Define training callbacks."""
    return [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=0)
    ]


def compute_class_weights(y_train: np.ndarray) -> dict:
    """Compute class weights to handle class imbalance."""
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    print(f"[MODEL] Class weights: {cw}")
    return cw


def train(X_train, X_val, y_train, y_val,
          learning_rate=1e-3, batch_size=64, epochs=100):
    """Build and train the model."""
    input_dim = X_train.shape[1]
    model = build_model(input_dim, learning_rate)
    model.summary()

    cw = compute_class_weights(y_train)
    ckpt_path = os.path.join(MODEL_DIR, "best_checkpoint.keras")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=cw,
        callbacks=get_callbacks(ckpt_path),
        verbose=1
    )
    return model, history


def evaluate(model: keras.Model, X_test: np.ndarray, y_test: np.ndarray,
             threshold: float = 0.5) -> dict:
    """Evaluate model on test set and return all metrics."""
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred).tolist()
    cr   = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])

    results = {
        "accuracy" : round(acc,  4),
        "precision": round(prec, 4),
        "recall"   : round(rec,  4),
        "f1_score" : round(f1,   4),
        "roc_auc"  : round(auc,  4),
        "confusion_matrix": cm,
        "threshold": threshold
    }

    print("\n" + "=" * 50)
    print("MODEL EVALUATION ON TEST SET")
    print("=" * 50)
    for k, v in results.items():
        if k != "confusion_matrix":
            print(f"  {k:<15}: {v}")
    print("\nClassification Report:")
    print(cr)

    return results, y_prob


def save_evaluation_charts(history, y_test, y_prob, results) -> None:
    """Save: training curves, confusion matrix, ROC curve."""

    # ── Training curves ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Model Training History", fontsize=14, fontweight="bold")

    axes[0].plot(history.history["loss"],     label="Train Loss",      color="#4A90D9")
    axes[0].plot(history.history["val_loss"], label="Validation Loss", color="#E74C3C")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"],     label="Train Acc",      color="#4A90D9")
    axes[1].plot(history.history["val_accuracy"], label="Validation Acc", color="#E74C3C")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training vs Validation Accuracy")
    axes[1].legend()

    path = os.path.join(CHART_DIR, "training_curves.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  [SAVED] {path}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = np.array(results["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns_cm = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(sns_cm)
    tick_marks = np.arange(2)
    ax.set_xticks(tick_marks); ax.set_xticklabels(["No Churn", "Churn"], fontsize=12)
    ax.set_yticks(tick_marks); ax.set_yticklabels(["No Churn", "Churn"], fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight="bold")
    path = os.path.join(CHART_DIR, "confusion_matrix.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  [SAVED] {path}")

    # ── ROC curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = results["roc_auc"]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#E74C3C", lw=2, label=f"ROC Curve (AUC = {auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#E74C3C")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    path = os.path.join(CHART_DIR, "roc_curve.png")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print(f"  [SAVED] {path}")


def run_model_pipeline() -> None:
    """Full pipeline: load splits -> train -> evaluate -> save."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # ── Train ──────────────────────────────────────────────────────────────────
    model, history = train(X_train, X_val, y_train, y_val,
                           learning_rate=1e-3, batch_size=64, epochs=100)

    # ── Evaluate ───────────────────────────────────────────────────────────────
    results, y_prob = evaluate(model, X_test, y_test)

    # ── Save model ─────────────────────────────────────────────────────────────
    model.save(MODEL_PATH)
    print(f"[MODEL] Saved model: {MODEL_PATH}")

    # ── Save results JSON ──────────────────────────────────────────────────────
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[MODEL] Saved evaluation results: {RESULTS_PATH}")

    # ── Save charts ────────────────────────────────────────────────────────────
    print("\n[MODEL] Saving evaluation charts...")
    save_evaluation_charts(history, y_test, y_prob, results)
    print("\n[MODEL] Done! All outputs saved.")


if __name__ == "__main__":
    run_model_pipeline()
