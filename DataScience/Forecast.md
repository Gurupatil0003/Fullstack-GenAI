```python
import streamlit as st
import pandas as pd
import plotly.express as px
from forecast_model import train_prophet
from prepare_data import load_and_merge_data
from mongo_utils import save_forecast  # ← Your module where save_forecast is defined

# Title and layout
st.set_page_config(layout="wide")
st.title("📦 Supply Chain Forecasting Dashboard")

# Load data
@st.cache_data
def load_data():
    return load_and_merge_data()

df = load_data()
store_ids = sorted(df['Store'].unique())
dept_ids = sorted(df['Dept'].unique())

# Sidebar filters
st.sidebar.header("🔍 Filter")
store_id = st.sidebar.selectbox("Select Store", store_ids)
dept_id = st.sidebar.selectbox("Select Department", dept_ids)

# Filter data
filtered = df[(df["Store"] == store_id) & (df["Dept"] == dept_id)].sort_values("Date")

# Sales trend chart
st.subheader(f"📈 Weekly Sales (Store {store_id} | Dept {dept_id})")
fig1 = px.line(filtered, x="Date", y="Weekly_Sales", title="Historical Weekly Sales")
st.plotly_chart(fig1, use_container_width=True)

# Forecast chart
st.subheader("🔮 Forecast: Next 90 Days")
forecast_df = train_prophet(store_id, dept_id)

# Plot forecast
fig2 = px.line(forecast_df, x="ds", y="yhat", title="Prophet Forecast")
fig2.add_scatter(x=forecast_df["ds"], y=forecast_df["yhat_upper"], mode="lines", name="Upper Bound")
fig2.add_scatter(x=forecast_df["ds"], y=forecast_df["yhat_lower"], mode="lines", name="Lower Bound")
st.plotly_chart(fig2, use_container_width=True)

# Save to MongoDB
if st.button("💾 Save Forecast to MongoDB"):
    forecast_data = forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")
    save_forecast(store_id, dept_id, forecast_data)
    st.success("Forecast saved to MongoDB!")

# Markdown analysis
st.subheader("📊 Average MarkDowns")
markdown_cols = [col for col in filtered.columns if "MarkDown" in col]
if markdown_cols:
    st.bar_chart(filtered[markdown_cols].mean())
else:
    st.info("No markdown data available for this store/department.")

st.caption("Built with ❤️ by Mounesh")


```

```python
from prophet import Prophet
import pandas as pd
from prepare_data import load_and_merge_data

def train_prophet(store_id=1, dept_id=1):
    df = load_and_merge_data()

    # Filter for a specific store & department
    df = df[(df["Store"] == store_id) & (df["Dept"] == dept_id)]

    # Rename columns for Prophet
    prophet_df = df[["Date", "Weekly_Sales"]].rename(columns={"Date": "ds", "Weekly_Sales": "y"})

    # Train the model
    model = Prophet()
    model.fit(prophet_df)

    # Forecast 90 days ahead
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10)



```

```python
import pandas as pd

def load_and_merge_data():
    # Load the datasets
    sales_df = pd.read_csv("train.csv", parse_dates=["Date"])
    features_df = pd.read_csv("features.csv", parse_dates=["Date"])
    stores_df = pd.read_csv("stores.csv")

    # Merge all data
    merged = pd.merge(sales_df, features_df, on=["Store", "Date"], how="left")
    merged = pd.merge(merged, stores_df, on="Store", how="left")

    # Fill NA with 0 for markdowns
    for i in range(1, 6):
        merged[f"MarkDown{i}"] = merged[f"MarkDown{i}"].fillna(0)

    return merged



```

```python
from forecast_model import train_prophet

forecast = train_prophet(store_id=1, dept_id=1)
print(forecast)



```

```python
from pymongo import MongoClient
from datetime import datetime

def convert_forecast_for_mongo(forecast_list):
    return [
        {
            "ds": item["ds"].to_pydatetime() if hasattr(item["ds"], "to_pydatetime") else item["ds"],
            "yhat": float(item["yhat"]),
            "yhat_lower": float(item["yhat_lower"]),
            "yhat_upper": float(item["yhat_upper"])
        }
        for item in forecast_list
    ]

def save_forecast(store_id, dept_id, forecast_data):
    client = MongoClient("mongodb://localhost:27017")
    db = client["forecast_db"]
    collection = db["sales_forecasts"]

    doc = {
        "store_id": int(store_id),
        "dept_id": int(dept_id),
        "forecast": convert_forecast_for_mongo(forecast_data),
        "timestamp": datetime.utcnow()
    }

    collection.insert_one(doc)


```
