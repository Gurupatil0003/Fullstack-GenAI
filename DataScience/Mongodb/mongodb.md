## Start App
```python
uvicorn a:app --reload


uvicorn app:app --reload

python -m uvicorn --version

python -m uvicorn app:app --reload

```

## database

```python
use salesdb
db.sales.insertMany([
  { "product": "Laptop", "quantity": 2, "price": 50000 },
  { "product": "Phone", "quantity": 5, "price": 15000 },
  { "product": "Tablet", "quantity": 3, "price": 20000 }
]) 

```

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

## Spark

```python

# Step 1: Install PySpark
!pip install pyspark

# Step 2: Import SparkSession
from pyspark.sql import SparkSession

# Create Spark Session
spark = SparkSession.builder.appName("ColabPySparkExample").getOrCreate()


# Step 4: Load CSV into PySpark DataFrame
df = spark.read.csv("employees.csv", header=True, inferSchema=True)

# Show original data
print("Original Data:")
df.show()

# Step 5: Basic Operations
print("Select name and salary:")



df.select("name", "salary").show()

print("Filter age > 30:")
df.filter(df.age > 30).show()

# Step 6: GroupBy and Aggregation
print("Average salary by department:")
df.groupBy("department").avg("salary").show()

# Step 7: Sort data by salary (descending)
print("Employees sorted by salary (High to Low):")
df.orderBy(df.salary.desc()).show()

# Step 8: Stop Spark
spark.stop()
```

## DBS
```python

# MongoDB Basic Commands Cheat Sheet

## 1. Start Mongo Shell
mongosh

## 2. Database Commands
show dbs                     # Show all databases
use myDatabase               # Switch / Create database
db                           # Show current database
db.dropDatabase()            # Drop current database

## 3. Collection Commands
show collections                         # Show collections
db.createCollection("myCollection")      # Create collection
db.myCollection.drop()                   # Drop collection

## 4. Insert Data
db.myCollection.insertOne({ name: "John", age: 25 })   # Insert one
db.myCollection.insertMany([
  { name: "Alice", age: 22 },
  { name: "Bob", age: 30 }
])                                                     # Insert many

## 5. Read Data
db.myCollection.find()                     # Find all
db.myCollection.find().pretty()            # Pretty print
db.myCollection.find({ age: 25 })          # Find by condition
db.myCollection.find({}, { name: 1 })      # Projection (select fields)
db.myCollection.find().sort({ age: 1 })    # Sort ascending
db.myCollection.find().sort({ age: -1 })   # Sort descending
db.myCollection.find().limit(5)            # Limit results
db.myCollection.find().skip(5)             # Skip records

## 6. Update Data
db.myCollection.updateOne(
  { name: "John" },
  { $set: { age: 30 } }
)
db.myCollection.updateMany(
  { age: { $lt: 25 } },
  { $set: { status: "minor" } }
)

## 7. Delete Data
db.myCollection.deleteOne({ name: "John" })
db.myCollection.deleteMany({ age: { $lt: 18 } })

## 8. Count Documents
db.myCollection.countDocuments()

## 9. Indexes
db.myCollection.createIndex({ name: 1 })
db.myCollection.getIndexes()
db.myCollection.dropIndex("name_1")

```

```python
show dbs                     # Show all databases
use myDatabase               # Switch / Create database
db 


show collections                         # Show collections
db.createCollection("myCollection")

db.myCollection.insertOne({ name: "John", age: 25 })   # Insert one

db.myCollection.find()                     # Find all

db.myCollection.find({ age: 25 })          # Find by condition

```

# Raw code
```python

from pymongo import MongoClient
import pandas as pd

client=MongoClient("mongodb://localhost:27017/")
db=client["salesdb"]
so=db["sales"]
ta=db["sa"]

data=list(so.find())
df=pd.DataFrame(data)
print(df)

if '_id' in df.columns:
    df=df.drop(columns=['_id'])

df["total"]=df['price']*df['quantity']
df=df.sort_values(by='total',ascending=False)

ta.delete_many({})
ta.insert_many(df.to_dict('records'))
print("_____________")
print(df)




```
