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

https://docs.google.com/spreadsheets/d/1_ixKQ-wIsOwJj8McX1Q6GgU4YV9bQuddkXmZxWWyelU/edit?usp=sharing

## app.py

```python
from flask import Flask, render_template, request, redirect, url_for, Response
from pymongo import MongoClient
import gridfs
from bson import ObjectId

app = Flask(__name__)

# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client['hotel_booking']
fs = gridfs.GridFS(db)
hotels = db['hotels']
bookings = db['bookings']

@app.route('/')
def home():
    # Show only hotels with images
    all_hotels = list(hotels.find({"image_id": {"$exists": True}}))
    return render_template('index.html', hotels=all_hotels)

@app.route('/image/<image_id>')
def get_image(image_id):
    image = fs.get(ObjectId(image_id))
    return Response(image.read(), mimetype='image/jpeg')
    

@app.route('/book/<hotel_id>', methods=['GET', 'POST'])
def book(hotel_id):
    hotel = hotels.find_one({"_id": ObjectId(hotel_id)})
    if request.method == 'POST':
        bookings.insert_one({
            "hotel_id": hotel_id,
            "name": request.form['name'],
            "checkin": request.form['checkin'],
            "checkout": request.form['checkout']
        })
        return redirect(url_for('home'))
    return render_template('booking.html', hotel=hotel)

@app.route('/add-hotel', methods=['GET', 'POST'])
def add_hotel():
    if request.method == 'POST':
        image = request.files['image']
        if image:
            image_id = fs.put(image, filename=image.filename)
            hotels.insert_one({
                "name": request.form['name'],
                "location": request.form['location'],
                "price": request.form['price'],
                "image_id": image_id
            })
        return redirect(url_for('home'))
    return render_template('add_hotel.html')

if __name__ == '__main__':
    app.run(debug=True)



```

## Index.html
```python
<!DOCTYPE html>
<html>
<head>
    <title>Hotels</title>
</head>
<body>
    <h1>Hotels</h1>
    <a href="{{ url_for('add_hotel') }}">Add Hotel</a>
    <hr>
    {% for hotel in hotels %}
        <div>
            <img src="{{ url_for('get_image', image_id=hotel.image_id) }}" width="150"><br>
            {{ hotel.name }} - {{ hotel.location }} - Rs.{{ hotel.price }}<br>
            <a href="{{ url_for('book', hotel_id=hotel._id) }}">Book</a>
        </div>
        <hr>
    {% endfor %}
</body>
</html>



```
# Booking Html

```python

<!DOCTYPE html>
<html>
<head>
    <title>Booking</title>
</head>
<body>
    <h2>Book {{ hotel.name }}</h2>
    <form method="POST">
        Name: <input type="text" name="name"><br><br>
        Check-in: <input type="date" name="checkin"><br><br>
        Check-out: <input type="date" name="checkout"><br><br>
        <input type="submit" value="Book">
    </form>
    <a href="/">Back</a>
</body>
</html>


```

## Add hotel html

```python
<!DOCTYPE html>
<html>
<head>
    <title>Add Hotel</title>
</head>
<body>
    <h2>Add Hotel</h2>
    <form method="POST" enctype="multipart/form-data">
        Name: <input type="text" name="name"><br><br>
        Location: <input type="text" name="location"><br><br>
        Price: <input type="number" name="price"><br><br>
        Image: <input type="file" name="image"><br><br>
        <input type="submit" value="Add">
    </form>
    <a href="/">Back</a>
</body>
</html>


```



## 📌 Animation URLs Table

| **Category**       | **Animation**            | **JSON URL** |
|--------------------|--------------------------|-------------|
| Loading           | Loading Spinner          | [Link](https://assets9.lottiefiles.com/packages/lf20_usmfx6bp.json) |
| Loading           | Loading Dots             | [Link](https://assets2.lottiefiles.com/packages/lf20_hdy0htc0.json) |
| Loading           | Loading Bar              | [Link](https://assets8.lottiefiles.com/packages/lf20_vfdkv6om.json) |
| Loading           | Circular Loader          | [Link](https://assets1.lottiefiles.com/packages/lf20_q5pk6p1k.json) |
| Loading           | Simple Loader            | [Link](https://assets10.lottiefiles.com/packages/lf20_bhw1ul4g.json) |
| Success           | Success Checkmark        | [Link](https://assets2.lottiefiles.com/packages/lf20_touohxv0.json) |
| Success           | Success + Confetti       | [Link](https://assets4.lottiefiles.com/packages/lf20_jbrw3hcz.json) |
| Success           | Done ✔ Animation         | [Link](https://assets7.lottiefiles.com/packages/lf20_jzq2az8g.json) |
| Error / Warning   | Error Cross              | [Link](https://assets2.lottiefiles.com/packages/lf20_qp1q7mct.json) |
| Error / Warning   | Alert Animation          | [Link](https://assets9.lottiefiles.com/packages/lf20_jz6g8znp.json) |
| Error / Warning   | Warning Sign             | [Link](https://assets8.lottiefiles.com/packages/lf20_kyu7xb1v.json) |
| AI / ML           | Data Analysis            | [Link](https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json) |
| AI / ML           | AI Brain                 | [Link](https://assets3.lottiefiles.com/packages/lf20_bdlrkrqv.json) |
| AI / ML           | Machine Learning Model   | [Link](https://assets10.lottiefiles.com/packages/lf20_dyqfnxau.json) |
| AI / ML           | Neural Network           | [Link](https://assets6.lottiefiles.com/packages/lf20_o7wz8d5x.json) |
| Coding / Tech     | Coding Animation         | [Link](https://assets1.lottiefiles.com/packages/lf20_gjmecwii.json) |
| Coding / Tech     | Programmer Working       | [Link](https://assets8.lottiefiles.com/packages/lf20_x62chJ.json) |
| Coding / Tech     | Developer Desk           | [Link](https://assets2.lottiefiles.com/packages/lf20_gigyrcoy.json) |
| Coding / Tech     | API Integration          | [Link](https://assets9.lottiefiles.com/packages/lf20_2ks7jvhv.json) |
| Coding / Tech     | Cloud Deployment         | [Link](https://assets3.lottiefiles.com/packages/lf20_zrqthn6o.json) |
| Fun / UI          | Party Confetti           | [Link](https://assets9.lottiefiles.com/packages/lf20_q5pk6p1k.json) |
| Fun / UI          | Heart Like               | [Link](https://assets10.lottiefiles.com/packages/lf20_xlkxtmul.json) |
| Fun / UI          | Fireworks                | [Link](https://assets4.lottiefiles.com/packages/lf20_vfdkv6om.json) |
| Fun / UI          | Celebration              | [Link](https://assets6.lottiefiles.com/packages/lf20_fyye8szy.json) |
| Fun / UI          | Emoji Reaction           | [Link](https://assets7.lottiefiles.com/packages/lf20_vgttfyaz.json) |
| Data Viz          | Bar Chart Animation      | [Link](https://assets10.lottiefiles.com/packages/lf20_7pvyv9sk.json) |
| Data Viz          | Pie Chart Animation      | [Link](https://assets2.lottiefiles.com/packages/lf20_t0cjqj.json) |
| Data Viz          | Line Graph Animation     | [Link](https://assets8.lottiefiles.com/packages/lf20_zrqthn6o.json) |
| Data Viz          | Dashboard Analytics      | [Link](https://assets1.lottiefiles.com/packages/lf20_ye3fehx0.json) |



```python

You are an intelligent, friendly, and highly skilled personal assistant who helps users manage their daily tasks, knowledge, productivity, and general inquiries. Respond in a professional, friendly, and concise tone. Use markdown formatting.

### 🧾 1. Task Understanding
- Clarify the user's goal or request.
- Rephrase it briefly in your own words.
- Ask for clarifications if needed.

### ✅ 2. Solution / Response
- Provide the most accurate and helpful response.
- Use bullet points or steps where appropriate.
- Offer suggestions, alternatives, or actions.
- If the request includes a topic or query, perform a DuckDuckGo search to:
  - Retrieve the most relevant, recent, and authoritative information.
  - Include 2–3 links with brief descriptions.

### 🧘 3. Friendly Summary
- End with a short, friendly message or suggestion for next steps.
- Keep the tone light, warm, and encouraging.

```

```python
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the patient's medical image and structure your response as follows:

### 1. Image Type & Region
- Specify imaging modality (X-ray/MRI/CT/Ultrasound/etc.)
- Identify the patient's anatomical region and positioning
- Comment on image quality and technical adequacy

### 2. Key Findings
- List primary observations systematically
- Note any abnormalities in the patient's imaging with precise descriptions
- Include measurements and densities where relevant
- Describe location, size, shape, and characteristics
- Rate severity: Normal/Mild/Moderate/Severe

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level
- List differential diagnoses in order of likelihood
- Support each diagnosis with observed evidence from the patient's imaging
- Note any critical or urgent findings

### 4. Patient-Friendly Explanation
- Explain the findings in simple, clear language that the patient can understand
- Avoid medical jargon or provide clear definitions
- Include visual analogies if helpful
- Address common patient concerns related to these findings

### 5. Research Context
IMPORTANT: Use the DuckDuckGo search tool to:
- Find recent medical literature about similar cases
- Search for standard treatment protocols
- Provide a list of relevant medical links of them too
- Research any relevant technological advances
- Include 2-3 key references to support your analysis

Format your response using clear markdown headers and bullet points. Be concise yet thorough

```


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
https://colab.research.google.com/drive/1j8K2wfECA16VFrIcvr0xbMNqr8hAKzi9?usp=sharing#scrollTo=TmSAurlmUnbJ


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
~~~python



import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D

# Config
IMG_SIZE = 64
BATCH_SIZE = 32

# Updated Paths for Brain Tumor Dataset
train_path = "/content/Brain-Tumor-Data-Set-main/Brain Tumor Data Set/Test"
val_path = "/content/Brain-Tumor-Data-Set-main/Brain Tumor Data Set/Train"

# Data Loaders
train_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_data = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary'
)

# MLP Model

# model = Sequential([
#     Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
#     MaxPooling2D(2,2),
#     Conv2D(64, (3,3), activation='relu'),
#     MaxPooling2D(2,2),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])


model = Sequential([
    Flatten(input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
model.fit(train_data, epochs=5, validation_data=val_data)


~~~
