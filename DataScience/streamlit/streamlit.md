```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import pandas as pd

# Load dataset
df = pd.read_csv("house_prices.csv")

# Features and target
X = df[['Area_sqft', 'Bedrooms', 'Bathrooms']]
y = df['Price']

# Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "house_price_model.pkl")
print("Model trained and saved as house_price_model.pkl")

import streamlit as st
import joblib

# Load model
model = joblib.load("house_price_model.pkl")

st.title("🏡 House Price Prediction")

# Input fields
area = st.number_input("Enter Area (sqft):", min_value=500, max_value=5000, step=100)
bedrooms = st.number_input("Enter Number of Bedrooms:", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Enter Number of Bathrooms:", min_value=1, max_value=5, step=1)

if st.button("Predict Price"):
    prediction = model.predict([[area, bedrooms, bathrooms]])[0]
    st.success(f"Estimated Price: ₹{prediction:,.2f}")
```
