# =============================================================================
# Logistic Regression from Scratch — Binary Classification
# UTS Machine Learning — Yaeesh Mahomed (24957692)
#
# Dataset:  Breast Cancer Wisconsin (scikit-learn)
# Task:     Binary classification — Malignant (0) vs Benign (1)
# Features: 30 numerical features, 569 samples
#
# Pipeline:
#   1. Feature standardisation
#   2. Custom LogisticRegression class (sigmoid, BCE + L2, gradient descent)
#   3. 5-fold cross-validated hyperparameter grid search
#   4. Final model evaluation (accuracy, ROC-AUC, precision, recall, F1, MCC)
#   5. Confusion matrix, training loss curve, ROC curve
#
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_curve, auc
)


# =============================================================================
# LOGISTIC REGRESSION CLASS
# =============================================================================

class LogisticRegression:
    """
    Binary logistic regression implemented from scratch using:
      - Sigmoid activation
      - Binary cross-entropy loss + L2 (Ridge) regularisation
      - Batch gradient descent
    """

    def __init__(self, learning_rate=0.1, max_iter=1000, l2_lambda=0.0):
        """
        Args:
            learning_rate: step size for gradient descent (alpha)
            max_iter:      number of gradient descent iterations
            l2_lambda:     L2 regularisation strength (lambda); 0 = no regularisation
        """
        self.lr          = learning_rate
        self.max_iter    = max_iter
        self.l2          = l2_lambda
        self.weights     = None
        self.bias        = 0
        self.loss_history = []

    # ------------------------------------------------------------------
    # Forward propagation
    # ------------------------------------------------------------------

    def sigmoid(self, z):
        """
        Sigmoid function: sigma(z) = 1 / (1 + e^-z)
        Clips z to [-500, 500] to prevent overflow in np.exp.
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def predict_proba(self, X):
        """
        Compute predicted probabilities: y_hat = sigma(X @ w + b)

        Returns:
            Array of probabilities in (0, 1) for each sample.
        """
        return self.sigmoid(X @ self.weights + self.bias)

    def predict(self, X, threshold=0.5):
        """
        Convert probabilities to binary class labels.

        Args:
            threshold: decision boundary (default 0.5)
                       lower to increase recall, raise to increase precision

        Returns:
            Integer array of 0s and 1s.
        """
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Loss function
    # ------------------------------------------------------------------

    def compute_loss(self, X, y):
        """
        Binary cross-entropy loss with L2 regularisation:

            L = -(1/m) * sum[ y*log(y_hat) + (1-y)*log(1-y_hat) ]
                + (lambda / 2m) * sum(w^2)

        The 1e-15 epsilon prevents log(0) errors.
        L2 is used (not L1) to preserve all feature contributions —
        important for medical diagnosis where every feature may matter.

        Returns:
            Scalar total loss value.
        """
        m          = len(y)
        proba      = self.predict_proba(X)
        bce_loss   = -(1 / m) * np.sum(
            y * np.log(proba + 1e-15) + (1 - y) * np.log(1 - proba + 1e-15)
        )
        l2_penalty = (self.l2 / (2 * m)) * np.sum(self.weights ** 2)
        return bce_loss + l2_penalty

    # ------------------------------------------------------------------
    # Backward propagation (gradient descent)
    # ------------------------------------------------------------------

    def gradient_descent(self, X, y):
        """
        Compute gradients and update weights and bias.

        Gradient derivation (via chain rule):
            dL/dw = (1/m) * X^T @ (y_hat - y)  +  (lambda/m) * w
            dL/db = (1/m) * sum(y_hat - y)

        Parameter update:
            w := w - alpha * dw
            b := b - alpha * db

        The L2 term (lambda/m)*w is the derivative of the L2 penalty,
        with the 1/2 cancelling out during differentiation.
        """
        m     = len(y)
        proba = self.predict_proba(X)

        # Gradient for weights — X.T combines prediction errors across all features
        dw = (1 / m) * X.T @ (proba - y) + (self.l2 / m) * self.weights
        # Gradient for bias — average prediction error (bias affects all predictions equally)
        db = (1 / m) * np.sum(proba - y)

        self.weights -= self.lr * dw
        self.bias    -= self.lr * db

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X, y):
        """
        Train the model by running gradient descent for max_iter iterations.

        Weights are initialised randomly (better exploration than zeros).
        Bias is initialised to 0.
        Loss is recorded at each iteration for visualisation.
        """
        n_features        = X.shape[1]
        self.weights      = np.random.randn(n_features)  # random init
        self.bias         = 0
        self.loss_history = []

        for _ in range(self.max_iter):
            loss = self.compute_loss(X, y)
            self.loss_history.append(loss)
            self.gradient_descent(X, y)


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

# Load Breast Cancer Wisconsin dataset
data    = load_breast_cancer()
X       = data.data
y       = data.target
print("Dataset shape :", X.shape)
print("Target shape  :", y.shape)
print("Classes       :", data.target_names)   # ['malignant' 'benign']

# Standardise features — ensures equal influence regardless of original scale
# (e.g. "mean area" ~1000s vs "smoothness" ~0.1)
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 80/20 train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, train_size=0.8, random_state=42
)
print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")


# =============================================================================
# HYPERPARAMETER TUNING — 5-FOLD CROSS VALIDATION
# =============================================================================

param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],
    "l2_lambda":     [0.0,  0.1,  1.0, 10.0],
    "max_iter":      [500, 1000],
}

best_params = None
best_score  = -np.inf                              # initialise below any real score
kf          = KFold(n_splits=5, shuffle=True, random_state=42)

print("\nRunning hyperparameter grid search...")

for lr in param_grid["learning_rate"]:
    for l2 in param_grid["l2_lambda"]:
        for mi in param_grid["max_iter"]:
            cv_scores = []

            for train_idx, val_idx in kf.split(X_train):
                X_tr,  X_val  = X_train[train_idx], X_train[val_idx]
                y_tr,  y_val  = y_train[train_idx], y_train[val_idx]

                model = LogisticRegression(
                    learning_rate=lr, max_iter=mi, l2_lambda=l2
                )
                model.fit(X_tr, y_tr)

                y_val_pred = model.predict(X_val)
                cv_scores.append(accuracy_score(y_val, y_val_pred))

            avg_score = np.mean(cv_scores)

            if avg_score > best_score:
                best_score  = avg_score
                best_params = {
                    "learning_rate": lr,
                    "l2_lambda":     l2,
                    "max_iter":      mi,
                }

print("\n=== Best Hyperparameters ===")
print(best_params, "  |  CV Accuracy:", round(best_score, 4))


# =============================================================================
# FINAL MODEL TRAINING
# =============================================================================

# Train on the full training set using the best hyperparameters
# (**best_params unpacks the dictionary as keyword arguments)
final_model = LogisticRegression(**best_params)
final_model.fit(X_train, y_train)


# =============================================================================
# EVALUATION
# =============================================================================

y_test_pred  = final_model.predict(X_test)
y_test_proba = final_model.predict_proba(X_test)

test_acc  = accuracy_score(y_test, y_test_pred)
roc_test  = roc_auc_score(y_test, y_test_proba)
precision = precision_score(y_test, y_test_pred)
recall    = recall_score(y_test, y_test_pred)
f1        = f1_score(y_test, y_test_pred)
mcc       = matthews_corrcoef(y_test, y_test_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
specificity    = tn / (tn + fp)

print("\n=== Test Set Results ===")
print(f"  Accuracy    : {test_acc:.4f}")
print(f"  ROC AUC     : {roc_test:.4f}")
print(f"  Precision   : {precision:.4f}")
print(f"  Recall      : {recall:.4f}")
print(f"  F1-Score    : {f1:.4f}")
print(f"  MCC         : {mcc:.4f}")
print(f"  Specificity : {specificity:.4f}")


# =============================================================================
# VISUALISATIONS
# =============================================================================

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap="Blues")
plt.colorbar()
plt.xticks([0, 1], ["Malignant", "Benign"])
plt.yticks([0, 1], ["Malignant", "Benign"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Test Set)")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="red", fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# --- Training Loss Curve ---
plt.figure(figsize=(7, 5))
plt.plot(final_model.loss_history, color="blue", lw=2)
plt.title("Training Loss Curve")
plt.xlabel("Iteration")
plt.ylabel("Binary Cross-Entropy Loss + L2 Penalty")
plt.grid(True)
plt.tight_layout()
plt.savefig("training_loss.png", dpi=300)
plt.show()

# --- ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc_val = auc(fpr, tpr)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="blue", lw=2,
         label=f"ROC curve (AUC = {roc_auc_val:.4f})")
plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test Set)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.show()
