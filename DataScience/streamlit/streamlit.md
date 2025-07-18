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


```python
import streamlit as st
import pandas as pd
import numpy as np

# ---------------------------
# PAGE CONFIGURATION
# ---------------------------
st.set_page_config(page_title="Single-Page Multi-Page App", layout="wide")

# ---------------------------
# SIDEBAR NAVIGATION
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📄 About", "📊 Charts", "📞 Contact"])

# ---------------------------
# HOME PAGE
# ---------------------------
if page == "🏠 Home":
    st.title("🏠 Home Page")
    st.write("Welcome to the **Home Page** of this single-page Streamlit app!")
    st.success("Use the sidebar to navigate to different sections.")

# ---------------------------
# ABOUT PAGE
# ---------------------------
elif page == "📄 About":
    st.title("📄 About Page")
    st.write("This is the About Page.")
    st.markdown("""
    **Features in this app:**
    - Sidebar Navigation
    - Multiple Pages in a Single File
    - Charts, Tables, and Forms
    """)

# ---------------------------
# CHARTS PAGE
# ---------------------------
elif page == "📊 Charts":
    st.title("📊 Charts Page")

    st.subheader("Random Data Chart")
    data = pd.DataFrame(np.random.randn(20, 3), columns=["A", "B", "C"])
    st.line_chart(data)

    st.subheader("Area Chart")
    st.area_chart(data)

    st.subheader("Data Table")
    st.table(data.head())

# ---------------------------
# CONTACT PAGE
# ---------------------------
elif page == "📞 Contact":
    st.title("📞 Contact Page")

    with st.form("contact_form"):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        message = st.text_area("Your Message")
        submitted = st.form_submit_button("Send")
        if submitted:
            st.success(f"Thank you {name}, we have received your message!")


```

```python
from streamlit_lottie import st_lottie
import requests

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


# Example Animation 1
lottie_animation = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_bdlrkrqv.json")
st_lottie(lottie_animation, height=300)


```

```python
https://assets2.lottiefiles.com/packages/lf20_touohxv0.json


```

```python

from streamlit_lottie import st_lottie
import requests

def load(url):
    r=requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()
lott=load("https://assets8.lottiefiles.com/packages/lf20_zrqthn6o.json")
st_lottie(lott,height=300)

```
