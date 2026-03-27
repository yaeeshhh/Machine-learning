# 🧬 Logistic Regression from Scratch — Breast Cancer Classification

> **UTS Machine Learning** | Yaeesh Mahomed (24957692)  
> Individual project — Logistic Regression implemented from scratch (Option A)

---

## 📌 Project Overview

This project implements a full **binary logistic regression classifier from scratch** using only NumPy, applied to the Breast Cancer Wisconsin dataset. Every component — the sigmoid function, binary cross-entropy loss, L2 regularisation, gradient descent, and k-fold cross-validation — is hand-coded without relying on scikit-learn's model implementations.

The goal was to deeply understand how the mathematical theory of logistic regression translates into working code, rather than simply calling a library function.

---

## 📓 Colab Notebook

🔗 [Open in Google Colab](https://colab.research.google.com/drive/1g_TZ4MbaBWmJsiaSwJdT6dyGikGdkrE?usp=sharing)

---

## 🗂️ Dataset

**Breast Cancer Wisconsin** — `sklearn.datasets.load_breast_cancer`

| Property | Value |
|---|---|
| Samples | 569 |
| Features | 30 numerical (mean radius, texture, smoothness, concavity, etc.) |
| Classes | 2 — Malignant (0), Benign (1) |
| Split | 80% train / 20% test |

---

## 🧮 Mathematical Foundation

### 1. Linear Combination

The model starts by computing a weighted sum of input features:

$$z = w^T x + b$$

where $x$ are the input features, $w$ are the learned weights, and $b$ is the bias term.

---

### 2. Sigmoid Function

Since $z$ can range from $-\infty$ to $+\infty$, the sigmoid function maps it to a probability in $(0, 1)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

This gives the hypothesis function:

$$\hat{y} = h(x) = \sigma(w^T x + b)$$

The prediction rule: classify as **1 (benign)** if $\hat{y} \geq 0.5$, else **0 (malignant)**.

---

### 3. Loss Function — Binary Cross-Entropy + L2 Regularisation

Binary cross-entropy (BCE) measures how far predicted probabilities are from true labels:

$$L_{BCE} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

To prevent overfitting, an **L2 (Ridge) regularisation** penalty is added:

$$L_2 = \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

The total loss becomes:

$$L = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} w_j^2$$

> **Why L2 and not L1?** L1 (Lasso) can shrink weights exactly to zero, effectively removing features. In cancer diagnosis, every measured feature may carry clinical significance — even small values can interact with other features to influence the outcome. L2 keeps all features active while still penalising large weights.

---

### 4. Gradient Descent — Derivation

To minimise the loss, gradients are computed via the chain rule and parameters are updated iteratively.

**Chain rule for weights:**

$$\frac{\partial L^{(i)}}{\partial w} = \frac{\partial L^{(i)}}{\partial \hat{y}^{(i)}} \cdot \frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial w}$$

**First factor** — sensitivity of loss to predicted probability:

$$\frac{\partial L^{(i)}}{\partial \hat{y}^{(i)}} = -\left( \frac{y^{(i)}}{\hat{y}^{(i)}} - \frac{1 - y^{(i)}}{1 - \hat{y}^{(i)}} \right)$$

**Second factor** — sensitivity of sigmoid output to linear score (derivative of sigmoid):

$$\frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}} = \hat{y}^{(i)}(1 - \hat{y}^{(i)})$$

Multiplying the first two factors simplifies cleanly to the prediction error:

$$\frac{\partial L^{(i)}}{\partial \hat{y}^{(i)}} \cdot \frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}} = \hat{y}^{(i)} - y^{(i)}$$

**Third factor:**

$$\frac{\partial z^{(i)}}{\partial w} = x^{(i)}$$

**Full gradient over all samples** (with L2 derivative, where the $\frac{1}{2}$ cancels):

$$\frac{\partial L}{\partial w} = \frac{1}{m} X^T (\hat{y} - y) + \frac{\lambda}{m} w$$

**Bias gradient** (bias derivative of $z$ with respect to $b$ is just 1, so it reduces to the average error):

$$\frac{\partial L}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

**Parameter updates:**

$$w := w - \alpha \cdot dw \qquad b := b - \alpha \cdot db$$

where $\alpha$ is the learning rate.

---

### 5. Evaluation Metrics

| Metric | Formula |
|---|---|
| Recall (Sensitivity) | $\dfrac{TP}{TP + FN}$ |
| Precision | $\dfrac{TP}{TP + FP}$ |
| F1-Score | $2 \times \dfrac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ |
| Specificity | $\dfrac{TN}{TN + FP}$ |
| MCC | $\dfrac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ |

---

## ⚙️ Training Pipeline

1. **Standardise features** — zero mean, unit variance: $x' = \dfrac{x - \mu}{\sigma}$
2. **Grid search** over hyperparameters with **5-fold cross-validation**
3. **Train final model** on full training set with best hyperparameters
4. **Evaluate** on held-out test set

### Hyperparameter Grid

| Parameter | Values Searched |
|---|---|
| `learning_rate` ($\alpha$) | 0.01, 0.05, 0.1 |
| `l2_lambda` ($\lambda$) | 0.0, 0.1, 1.0, 10.0 |
| `max_iter` | 500, 1000 |

**Best found:** `learning_rate=0.1`, `l2_lambda=0.1`, `max_iter=1000` → CV Accuracy: **0.9758**

---

## 📊 Results

| Metric | Score |
|---|---|
| Accuracy | 0.9474 |
| ROC AUC | 0.9918 |
| Precision | 0.9595 |
| Recall | 0.9667 |
| F1-Score | 0.9631 |
| MCC | 0.8896 |
| Specificity | 0.9231 |

The ROC-AUC of **0.9918** and the training loss converging smoothly to a minimum confirm the model trains correctly and generalises well to unseen data.

---

## 🚀 Running the Code

```bash
pip install -r requirements.txt
python main.py
```

Outputs saved: `confusion_matrix.png`, `training_loss.png`, `roc_curve.png`

---

## 🛠️ Tech Stack

`Python` `NumPy` `Matplotlib` `Scikit-learn (data loading + metrics only)`

---

## 📄 License

Submitted as academic coursework at UTS. Shared for portfolio and reference purposes only.
