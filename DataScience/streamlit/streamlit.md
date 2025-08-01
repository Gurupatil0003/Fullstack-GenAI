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

```python
import streamlit as st
import pandas as pd
import joblib
import datetime

st.set_page_config(page_title="Energy Predictor", page_icon="⚡")
st.title("⚡ Energy Consumption Predictor")

# Load model
model = joblib.load("Random_forest_model (2).pkl")

# Manually define the 29 expected features (edit this list to match your training set)
model_columns = [
    'num_occupants',
    'house_size_sqft',
    'monthly_income',
    'outside_temp_celsius',
    'year',
    'month',
    'day',
    'season',
    'heating_type_Electric',
    'heating_type_Gas',
    'heating_type_None',
    'cooling_type_AC',
    'cooling_type_Fan',
    'cooling_type_None',
    'manual_override_Y',
    'manual_override_N',
    'is_weekend',
    'temp_above_avg',
    'income_per_person',
    'square_feet_per_person',
    'high_income_flag',
    'low_temp_flag',
    'season_spring',
    'season_summer',
    'season_fall',
    'season_winter',
    'day_of_week_0',
    'day_of_week_6',
    'energy_star_home'
]

# User inputs
st.header("📝 Enter Input Data")

num_occupants = st.number_input("Number of Occupants", min_value=1, value=3)
house_size_sqft = st.number_input("House Size (sqft)", min_value=100, value=1200)
monthly_income = st.number_input("Monthly Income", min_value=1000, value=30000)
outside_temp_celsius = st.number_input("Outside Temp (°C)", value=25.0)
year = st.number_input("Year", min_value=2000, max_value=2100, value=2024)
month = st.number_input("Month", min_value=1, max_value=12, value=7)
day = st.number_input("Day", min_value=1, max_value=31, value=15)

# Validate date (no try-except used)
date_obj = datetime.date(year, month, day)
day_of_week = date_obj.weekday()

# Season detection
if month in [12, 1, 2]:
    season_label = 'winter'
elif month in [3, 4, 5]:
    season_label = 'spring'
elif month in [6, 7, 8]:
    season_label = 'summer'
else:
    season_label = 'fall'

# Binary options
heating_type = st.selectbox("Heating Type", ["Electric", "Gas", "None"])
cooling_type = st.selectbox("Cooling Type", ["AC", "Fan", "None"])
manual_override = st.radio("Manual Override", ["Yes", "No"])
energy_star_home = st.checkbox("Energy Star Certified Home", value=False)

# Derived features
is_weekend = int(day_of_week >= 5)
temp_above_avg = int(outside_temp_celsius > 22)
income_per_person = monthly_income / num_occupants
square_feet_per_person = house_size_sqft / num_occupants
high_income_flag = int(monthly_income > 40000)
low_temp_flag = int(outside_temp_celsius < 15)

# Input data dictionary
input_data = {
    'num_occupants': num_occupants,
    'house_size_sqft': house_size_sqft,
    'monthly_income': monthly_income,
    'outside_temp_celsius': outside_temp_celsius,
    'year': year,
    'month': month,
    'day': day,
    'season': {'spring': 2, 'summer': 3, 'fall': 4, 'winter': 1}[season_label],
    'heating_type_Electric': int(heating_type == "Electric"),
    'heating_type_Gas': int(heating_type == "Gas"),
    'heating_type_None': int(heating_type == "None"),
    'cooling_type_AC': int(cooling_type == "AC"),
    'cooling_type_Fan': int(cooling_type == "Fan"),
    'cooling_type_None': int(cooling_type == "None"),
    'manual_override_Y': int(manual_override == "Yes"),
    'manual_override_N': int(manual_override == "No"),
    'is_weekend': is_weekend,
    'temp_above_avg': temp_above_avg,
    'income_per_person': income_per_person,
    'square_feet_per_person': square_feet_per_person,
    'high_income_flag': high_income_flag,
    'low_temp_flag': low_temp_flag,
    'season_spring': int(season_label == "spring"),
    'season_summer': int(season_label == "summer"),
    'season_fall': int(season_label == "fall"),
    'season_winter': int(season_label == "winter"),
    'day_of_week_0': int(day_of_week == 0),
    'day_of_week_6': int(day_of_week == 6),
    'energy_star_home': int(energy_star_home)
}

# Build DataFrame
input_df = pd.DataFrame([input_data])

# # Ensure all expected features are present
# for col in model_columns:
#     if col not in input_df.columns:
#         input_df[col] = 0

# Reorder columns to match model
input_df = input_df[model_columns]

# Predict
if st.button("🔮 Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"🔋 Predicted Energy Consumption: *{prediction:.2f} kWh*")


```
