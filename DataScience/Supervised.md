# Datascience Notebook Examples

| Topic            | PDF Link                                                                                                                                     | Streamlit App                                                                                      | Colab Notebook                                                                                                                                           |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Classification     | <a href="PDF_LINK_HERE" target="_parent"><img src="https://img.shields.io/badge/Open in PDF-%23FF0000.svg?style=flat-square&logo=adobe&logoColor=white"/></a> | <a href="STREAMLIT_LINK_HERE" target="_parent"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/></a> | <a href="https://colab.research.google.com/drive/1F3z64bjBCmw7qmjAmUrtt0a4xo-7k_hs?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Regression    | <a href="PDF_LINK_HERE" target="_parent"><img src="https://img.shields.io/badge/Open in PDF-%23FF0000.svg?style=flat-square&logo=adobe&logoColor=white"/></a> | <a href="STREAMLIT_LINK_HERE" target="_parent"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/></a> | <a href="https://colab.research.google.com/drive/1nePFBkd0SOFDsTUaGo7c2vM-D2O3Q4F-?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Churn     | <a href="PDF_LINK_HERE" target="_parent"><img src="https://img.shields.io/badge/Open in PDF-%23FF0000.svg?style=flat-square&logo=adobe&logoColor=white"/></a> | <a href="STREAMLIT_LINK_HERE" target="_parent"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/></a> | <a href="https://colab.research.google.com/drive/1FMrPMla0SNmgYfAsP9hcx6Llu98u98-U?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Matplotlib    | <a href="PDF_LINK_HERE" target="_parent"><img src="https://img.shields.io/badge/Open in PDF-%23FF0000.svg?style=flat-square&logo=adobe&logoColor=white"/></a> | <a href="STREAMLIT_LINK_HERE" target="_parent"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/></a> | <a href="https://colab.research.google.com/drive/1STdP8lBpbyREeiPmTlXWMZPMCI6gjchC?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Plotly     | <a href="PDF_LINK_HERE" target="_parent"><img src="https://img.shields.io/badge/Open in PDF-%23FF0000.svg?style=flat-square&logo=adobe&logoColor=white"/></a> | <a href="STREAMLIT_LINK_HERE" target="_parent"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/></a> | <a href="https://colab.research.google.com/drive/1USNV4joQrFp81fvP__T4-9W7S5nhm9NA?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Rent_car     | <a href="PDF_LINK_HERE" target="_parent"><img src="https://img.shields.io/badge/Open in PDF-%23FF0000.svg?style=flat-square&logo=adobe&logoColor=white"/></a> | <a href="STREAMLIT_LINK_HERE" target="_parent"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/></a> | <a href="https://colab.research.google.com/drive/1RLiZqCfhawULkwLyz92AwtkBrJEkdhEu?usp=sharing#scrollTo=mqyiuJReow1E" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="https://colab.research.google.com/drive/1FTdJ8t7mgirUClfPMMOPOAt6m1tZesRh?usp=sharing"/></a> |





# 📘 Supervised Learning – Complete Guide

Supervised learning is a type of machine learning where the algorithm is trained on a **labeled dataset** — meaning each training example is paired with an output label. The goal is to **learn a function that maps inputs to outputs**.

---

## 🧠 Supervised Learning Workflow

```
Input Features (X) --> Model --> Predicts Output (Y_hat)
Target Labels (Y) --> Used during training
Loss Function --> Measures error between Y and Y_hat
Optimization --> Minimizes loss by updating model parameters
```

---

## 🧩 Common Supervised Learning Models

### 1. 📈 Linear Regression
- **Type:** Regression  
- **Definition:** Predicts a continuous output using a linear combination of input features.  
- **Equation:**  
  \[
  y = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n
  \]  
- **Loss Function:** Mean Squared Error (MSE):  
  \[
  MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  \]  
- **Use When:** Relationship is linear; model needs to be interpretable.

---

### 2. 📉 Logistic Regression
- **Type:** Classification  
- **Definition:** Predicts binary/multiclass outcomes using a logistic function.  
- **Equation:**  
  \[
  P(y=1|x) = \frac{1}{1 + e^{-(w^Tx + b)}}
  \]  
- **Loss Function:** Binary Cross-Entropy  
- **Use When:** Binary classification, probabilistic interpretation.

---

### 3. 🌳 Decision Tree
- **Type:** Classification / Regression  
- **Definition:** Splits data by feature thresholds to form a tree.  
- **Split Criteria:**
  - Gini: \( G = 1 - \sum p_i^2 \)
  - Entropy: \( H = -\sum p_i \log_2(p_i) \)
- **Use When:** Easy-to-interpret rules and non-linear splits are needed.

---

### 4. 🌲 Random Forest
- **Type:** Classification / Regression  
- **Definition:** Ensemble of decision trees for improved accuracy and reduced overfitting.  
- **Prediction:** Majority vote (classification), average (regression)  
- **Use When:** High accuracy is required; data is noisy.

---

### 5. 💠 Support Vector Machine (SVM)
- **Type:** Classification / Regression  
- **Definition:** Finds hyperplane that best separates classes with maximum margin.  
- **Objective:**  
  \[
  \max \frac{2}{||w||}
  \]  
- **Use When:** High-dimensional, small datasets.

---

### 6. 👥 k-Nearest Neighbors (k-NN)
- **Type:** Classification / Regression  
- **Definition:** Predicts by majority vote from k closest training points.  
- **Distance:**  
  - Euclidean:  
    \[
    d(x, x') = \sqrt{\sum (x_i - x'_i)^2}
    \]
- **Use When:** Small dataset, simple logic needed.

---

### 7. 🧮 Naive Bayes
- **Type:** Classification  
- **Definition:** Probabilistic model using Bayes' theorem with feature independence.  
- **Equation:**  
  \[
  P(y|x_1, ..., x_n) \propto P(y) \prod_{i=1}^{n} P(x_i|y)
  \]  
- **Use When:** Text classification (spam, sentiment), high-dimensional input.

---

### 8. 🚀 Gradient Boosting (e.g., XGBoost, LightGBM)
- **Type:** Classification / Regression  
- **Definition:** Sequentially builds trees that correct previous errors.  
- **Equation:**  
  \[
  \hat{y} = \sum_{m=1}^{M} \gamma_m h_m(x)
  \]  
- **Use When:** High accuracy, structured/tabular data.

---

### 9. 🧠 Artificial Neural Networks (ANN)
- **Type:** Classification / Regression  
- **Definition:** Mimics the human brain using layers of neurons.  
- **Forward Pass:**  
  \[
  a^{(l)} = f(W^{(l)}a^{(l-1)} + b^{(l)})
  \]  
- **Backpropagation:** Minimizes loss with gradient descent  
- **Use When:** Complex, non-linear data and large datasets.

---

## ✅ When and Why to Use Supervised Models:

| Scenario                        | Model Suggestion           | Reason                                  |
|---------------------------------|----------------------------|------------------------------------------|
| Predicting housing prices       | Linear Regression          | Continuous output, linear relationship   |
| Email spam detection            | Naive Bayes, Logistic Reg. | Fast and works well with text            |
| Disease diagnosis               | Random Forest, SVM         | Robust, handles non-linearity well       |
| Customer churn prediction       | XGBoost, Logistic Reg.     | Accuracy and interpretability            |
| Image classification (basic)   | ANN, SVM                   | Scalable and handles complex patterns    |
| Product recommendation (simple)| k-NN                       | Instance-based similarity                |
| Sentiment analysis              | Naive Bayes, Logistic Reg. | Categorical labels, text-based           |

---

## 📏 Evaluation Metrics

- **Classification:**
  - Accuracy
  - Precision / Recall / F1 Score
  - Confusion Matrix
  - ROC-AUC

- **Regression:**
  - R² Score
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)

---

> 🔎 Supervised Learning empowers machines to predict outcomes by learning from labeled examples. It’s your go-to for tasks like classification, regression, and real-world decision-making.
