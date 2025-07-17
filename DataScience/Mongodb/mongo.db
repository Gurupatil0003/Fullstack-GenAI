

## ETL with MongoDb
```python

"""
ETL Pipeline using existing MongoDB data
DB: salesdb
Collection: sales
"""

# -----------------------------
# 1. Import Libraries
# -----------------------------
from pymongo import MongoClient
import pandas as pd

# -----------------------------
# 2. Connect to MongoDB
# -----------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["salesdb"]           # Use your DB
source_collection = db["sales"]  # Existing collection
target_collection = db["sales_transformed"]

# -----------------------------
# 3. Extract: Fetch Data from MongoDB
# -----------------------------
data = list(source_collection.find())
df = pd.DataFrame(data)
print("\n✅ Extracted Data:")
print(df)

# -----------------------------
# 4. Transform: Clean and Add Fields
# -----------------------------
# Remove _id column
if '_id' in df.columns:
    df = df.drop(columns=['_id'])

# Add calculated field: total_amount = price * quantity
df['total_amount'] = df['price'] * df['quantity']

# Sort by total_amount descending
df = df.sort_values(by='total_amount', ascending=False)

print("\n✅ Transformed Data:")
print(df)

# -----------------------------
# 5. Load: Insert into Another Collection
# -----------------------------
# Clear old data in transformed collection
target_collection.delete_many({})
target_collection.insert_many(df.to_dict('records'))
print("\n✅ Loaded transformed data into 'sales_transformed' collection")

# -----------------------------
# 6. (Optional) Save to CSV
# -----------------------------
df.to_csv("sales_transformed.csv", index=False)
print("\n✅ Data saved to 'sales_transformed.csv'")

  
```


## Api Integration

```python

from fastapi import FastAPI
from pymongo import MongoClient
import pandas as pd
from typing import List

app = FastAPI()

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["salesdb"]
source_collection = db["sales"]
target_collection = db["sales_transformed"]

# ETL Function
def run_etl():
    # Extract
    data = list(source_collection.find())
    df = pd.DataFrame(data)

    if '_id' in df.columns:
        df = df.drop(columns=['_id'])

    # Transform
    df['total_amount'] = df['price'] * df['quantity']
    df = df.sort_values(by='total_amount', ascending=False)

    # Load
    target_collection.delete_many({})
    target_collection.insert_many(df.to_dict('records'))

    # Save CSV
    df.to_csv("sales_transformed.csv", index=False)

    return df.to_dict('records')

# API Endpoints
@app.get("/")
def home():
    return {"message": "ETL API is running!"}

@app.get("/run-etl")
@app.post("/run-etl")
def run_etl_api():
    transformed_data = run_etl()
    return {"message": "ETL process completed", "records": transformed_data[:5]}
```
