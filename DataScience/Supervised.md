# Datascience Notebook Examples

| Topic            | PDF Link                                                                                                                                     | Streamlit App                                                                                      | Colab Notebook                                                                                                                                           |
|------------------|----------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Classification     | <a href="PDF_LINK_HERE" target="_parent"><img src="https://img.shields.io/badge/Open in PDF-%23FF0000.svg?style=flat-square&logo=adobe&logoColor=white"/></a> | <a href="STREAMLIT_LINK_HERE" target="_parent"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/></a> | <a href="https://colab.research.google.com/drive/1F3z64bjBCmw7qmjAmUrtt0a4xo-7k_hs?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Regression    | <a href="PDF_LINK_HERE" target="_parent"><img src="https://img.shields.io/badge/Open in PDF-%23FF0000.svg?style=flat-square&logo=adobe&logoColor=white"/></a> | <a href="STREAMLIT_LINK_HERE" target="_parent"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/></a> | <a href="https://colab.research.google.com/drive/1nePFBkd0SOFDsTUaGo7c2vM-D2O3Q4F-?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Churn     | <a href="PDF_LINK_HERE" target="_parent"><img src="https://img.shields.io/badge/Open in PDF-%23FF0000.svg?style=flat-square&logo=adobe&logoColor=white"/></a> | <a href="STREAMLIT_LINK_HERE" target="_parent"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/></a> | <a href="https://colab.research.google.com/drive/1FMrPMla0SNmgYfAsP9hcx6Llu98u98-U?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Matplotlib    | <a href="PDF_LINK_HERE" target="_parent"><img src="https://img.shields.io/badge/Open in PDF-%23FF0000.svg?style=flat-square&logo=adobe&logoColor=white"/></a> | <a href="STREAMLIT_LINK_HERE" target="_parent"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/></a> | <a href="https://colab.research.google.com/drive/1STdP8lBpbyREeiPmTlXWMZPMCI6gjchC?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Plotly     | <a href="PDF_LINK_HERE" target="_parent"><img src="https://img.shields.io/badge/Open in PDF-%23FF0000.svg?style=flat-square&logo=adobe&logoColor=white"/></a> | <a href="STREAMLIT_LINK_HERE" target="_parent"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/></a> | <a href="https://colab.research.google.com/drive/1USNV4joQrFp81fvP__T4-9W7S5nhm9NA?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| Rent_car     | <a href="PDF_LINK_HERE" target="_parent"><img src="https://img.shields.io/badge/Open in PDF-%23FF0000.svg?style=flat-square&logo=adobe&logoColor=white"/></a> | <a href="STREAMLIT_LINK_HERE" target="_parent"><img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"/></a> | <a href="https://colab.research.google.com/drive/1RLiZqCfhawULkwLyz92AwtkBrJEkdhEu?usp=sharing#scrollTo=mqyiuJReow1E" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="https://colab.research.google.com/drive/1FTdJ8t7mgirUClfPMMOPOAt6m1tZesRh?usp=sharing"/></a> |




📌 What is Supervised Learning?
Definition:
Supervised learning is a type of machine learning where the algorithm is trained on a labeled dataset — meaning each training example is paired with an output label. The goal is to learn a function that maps inputs to outputs.

🧠 Supervised Learning Workflow:
Input Features (X) → Passed to the model

Target Labels (Y) → Used to guide learning

Model → Learns mapping f(X) ≈ Y

Prediction → Model outputs predicted labels for unseen data

Loss Function → Measures prediction error

Optimization → Adjust model to minimize the loss

🧩 Common Supervised Learning Models:
Let’s go one by one:

1. Linear Regression
🔹 Type: Regression

Definition:
Predicts a continuous output using a linear combination of input features.

Mathematics:

𝑦
=
𝑤
0
+
𝑤
1
𝑥
1
+
𝑤
2
𝑥
2
+
⋯
+
𝑤
𝑛
𝑥
𝑛
y=w 
0
​
 +w 
1
​
 x 
1
​
 +w 
2
​
 x 
2
​
 +⋯+w 
n
​
 x 
n
​
 
Where:

𝑦
y = predicted output

𝑤
𝑖
w 
i
​
  = learned weights

Loss Function:
Mean Squared Error (MSE):

MSE
=
1
𝑛
∑
𝑖
=
1
𝑛
(
𝑦
𝑖
−
𝑦
^
𝑖
)
2
MSE= 
n
1
​
  
i=1
∑
n
​
 (y 
i
​
 − 
y
^
​
  
i
​
 ) 
2
 
Prediction Logic:
Draws a straight line (or hyperplane) to best fit the training data.

Use When:

Relationship between variables is linear

You want interpretable models

2. Logistic Regression
🔹 Type: Classification (Binary/Multiclass)

Definition:
Used to predict categorical outcomes (like Yes/No, Spam/Not Spam).

Mathematics:

𝑃
(
𝑦
=
1
∣
𝑥
)
=
𝜎
(
𝑤
𝑇
𝑥
+
𝑏
)
=
1
1
+
𝑒
−
(
𝑤
𝑇
𝑥
+
𝑏
)
P(y=1∣x)=σ(w 
T
 x+b)= 
1+e 
−(w 
T
 x+b)
 
1
​
 
Loss Function:
Binary Cross-Entropy:

−
1
𝑛
∑
𝑖
=
1
𝑛
[
𝑦
𝑖
log
⁡
(
𝑦
^
𝑖
)
+
(
1
−
𝑦
𝑖
)
log
⁡
(
1
−
𝑦
^
𝑖
)
]
− 
n
1
​
  
i=1
∑
n
​
 [y 
i
​
 log( 
y
^
​
  
i
​
 )+(1−y 
i
​
 )log(1− 
y
^
​
  
i
​
 )]
Prediction Logic:
Uses sigmoid function to squash output into probability range [0, 1].

Use When:

You have a binary classification task

You want a fast, simple model with probabilistic output

3. Decision Tree
🔹 Type: Classification / Regression

Definition:
Tree-based structure that splits data based on feature thresholds to make decisions.

Mathematics:
Splits are made using metrics like:

Gini Impurity: 
𝐺
=
1
−
∑
𝑝
𝑖
2
G=1−∑p 
i
2
​
 

Entropy: 
𝐻
=
−
∑
𝑝
𝑖
log
⁡
2
(
𝑝
𝑖
)
H=−∑p 
i
​
 log 
2
​
 (p 
i
​
 )

MSE (for regression)

Prediction Logic:
Follows decision nodes until it reaches a leaf node (label).

Use When:

Data has clear decision rules

You want interpretability and non-linearity

4. Random Forest
🔹 Type: Classification / Regression
🔹 Ensemble of Decision Trees

Definition:
Combines multiple decision trees to improve accuracy and reduce overfitting.

Mathematics:

Aggregates outputs from multiple trees (majority vote for classification, average for regression)

Prediction Logic:
Each tree votes; result is average or mode.

Use When:

You want high accuracy

You need robustness to overfitting

5. Support Vector Machine (SVM)
🔹 Type: Classification / Regression (SVR)

Definition:
Finds the optimal hyperplane that maximally separates data into classes.

Mathematics:

Maximize margin 
⇒
2
∣
∣
𝑤
∣
∣
Maximize margin ⇒ 
∣∣w∣∣
2
​
 
Subject to:

𝑦
𝑖
(
𝑤
⋅
𝑥
𝑖
+
𝑏
)
≥
1
y 
i
​
 (w⋅x 
i
​
 +b)≥1
Kernel Trick: Allows non-linear separation using functions like RBF, Polynomial.

Prediction Logic:
Classifies based on which side of the hyperplane the data lies.

Use When:

You need robust classifier with small datasets

High-dimensional data

6. k-Nearest Neighbors (k-NN)
🔹 Type: Classification / Regression

Definition:
Instance-based method; classifies data points based on the majority vote of k nearest neighbors.

Mathematics:
Distance metrics:

Euclidean: 
𝑑
(
𝑥
,
𝑥
′
)
=
∑
(
𝑥
𝑖
−
𝑥
𝑖
′
)
2
d(x,x 
′
 )= 
∑(x 
i
​
 −x 
i
′
​
 ) 
2
 
​
 

Manhattan, Cosine, etc.

Prediction Logic:
No training; during prediction, checks nearest neighbors in training data.

Use When:

Data is small

You want a simple, non-parametric model

7. Naive Bayes
🔹 Type: Classification

Definition:
Probabilistic model based on Bayes’ theorem assuming feature independence.

Mathematics:

𝑃
(
𝑦
∣
𝑥
1
,
.
.
.
,
𝑥
𝑛
)
∝
𝑃
(
𝑦
)
∏
𝑖
=
1
𝑛
𝑃
(
𝑥
𝑖
∣
𝑦
)
P(y∣x 
1
​
 ,...,x 
n
​
 )∝P(y) 
i=1
∏
n
​
 P(x 
i
​
 ∣y)
Prediction Logic:
Chooses class with highest posterior probability.

Use When:

Text classification (spam detection)

Data is high-dimensional and categorical

8. Gradient Boosting (XGBoost, LightGBM, etc.)
🔹 Type: Classification / Regression
🔹 Ensemble of Trees

Definition:
Builds trees sequentially, each correcting the errors of the previous.

Mathematics:
Minimizes loss:

Prediction 
=
∑
𝑚
=
1
𝑀
𝛾
𝑚
ℎ
𝑚
(
𝑥
)
Prediction = 
m=1
∑
M
​
 γ 
m
​
 h 
m
​
 (x)
where 
ℎ
𝑚
h 
m
​
  is the m-th weak learner (typically a decision tree)

Loss Functions:

MSE (regression)

Log loss (classification)

Prediction Logic:
Each tree corrects the residuals of the last.

Use When:

You want state-of-the-art performance

You need to handle structured/tabular data

9. Artificial Neural Networks (ANN)
🔹 Type: Classification / Regression

Definition:
Mimics the human brain using layers of neurons to learn complex functions.

Mathematics:
Forward pass:

𝑎
(
𝑙
)
=
𝑓
(
𝑊
(
𝑙
)
𝑎
(
𝑙
−
1
)
+
𝑏
(
𝑙
)
)
a 
(l)
 =f(W 
(l)
 a 
(l−1)
 +b 
(l)
 )
Loss minimized via backpropagation using gradient descent.

Prediction Logic:
Activations flow forward; loss is backpropagated to update weights.

Use When:

Data is non-linear and large-scale

You want flexibility and deep architectures

✅ When and Why to Use Supervised Models:
Scenario	Model Suggestion	Reason
Predicting housing prices	Linear Regression	Continuous output, linear
Email spam detection	Naive Bayes, Logistic Regression	Text classification
Disease diagnosis	Random Forest, SVM	Non-linear, robust models
Customer churn prediction	XGBoost, Logistic Regression	Accuracy + Interpretability
Image classification (basic)	ANN, SVM	Non-linear, scalable
Product recommendation (simple)	k-NN	Memory-based similarity
Sentiment analysis	Naive Bayes, Logistic Regression	Text with categorical labels

🔧 Bonus: Evaluation Metrics
Accuracy

Precision / Recall / F1 Score

ROC-AUC

Confusion Matrix

R² Score (Regression)

