# Fullstack-GenAI

```python
import gradio as gr

# Predict function
def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_data)[0]
    return f"Predicted species: {prediction}"

# Gradio UI
iface = gr.Interface(
    fn=predict_iris,
    inputs=[
        gr.Number(label="Sepal Length (cm)"),
        gr.Number(label="Sepal Width (cm)"),
        gr.Number(label="Petal Length (cm)"),
        gr.Number(label="Petal Width (cm)")
    ],
    outputs="text",
    title="Iris Flower Species Predictor",
    description="Enter measurements to predict the species of Iris flower."
)

iface.launch()



```


```

# 📊 Data handling
import pandas as pd
import numpy as np

# 🔄 Preprocessing
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 🔀 Train-test split
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# 🧠 Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# ✅ Evaluation Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, RocCurveDisplay
)

# 📈 Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# ⚠️ Warnings
import warnings
warnings.filterwarnings('ignore')


# Uncomment these if you're using XGBoost or LightGBM
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier


```
