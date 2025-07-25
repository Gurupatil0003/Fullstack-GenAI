```python
import pandas as pd
from pymongo import MongoClient

# Step 1: Load the dataset
df = pd.read_csv('fake_job_postings.csv')

# Step 2: Clean (optional)
df.dropna(subset=['title', 'description', 'fraudulent'], inplace=True)

# Step 3: Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Adjust if using MongoDB Atlas
db = client['job_data']
collection = db['job_postings']

# Step 4: Insert into MongoDB
data_dict = df.to_dict(orient='records')
collection.insert_many(data_dict)

print("✅ Data inserted successfully into MongoDB!") 

```

```python

# train_model.py
import pandas as pd
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load from MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['job_data']
collection = db['job_postings']
data = list(collection.find())

# Prepare data
df = pd.DataFrame(data)
df = df[['description', 'fraudulent']].dropna()
X = df['description']
y = df['fraudulent'].astype(int)

# TF-IDF + Logistic Regression
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X_vec = vectorizer.fit_transform(X)
model = LogisticRegression()
model.fit(X_vec, y)

# Save model and vectorizer
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("✅ Model and vectorizer saved!")


```

```python

from flask import Flask, render_template, request
from pymongo import MongoClient
import joblib

# Load model and vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['job_data']
collection = db['predictions']

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    description = request.form['description']
    desc_vec = vectorizer.transform([description])
    prediction = model.predict(desc_vec)[0]
    result = "Fraudulent" if prediction == 1 else "Legitimate"

    # Save to MongoDB
    collection.insert_one({"description": description, "prediction": result})

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)


```

```python

<!DOCTYPE html>
<html>
<head><title>Job Fraud Detector</title></head>
<body style="text-align:center; font-family:sans-serif; padding:30px;">
    <h2>Job Fraud Detector</h2>
    <form method="post" action="/predict">
        <textarea name="description" rows="10" cols="60" placeholder="Enter job description..." required></textarea><br><br>
        <button type="submit">Check</button>
    </form>
    {% if prediction %}
        <h3>Prediction: {{ prediction }}</h3>
    {% endif %}
</body>
</html>


```


```python

from flask import Flask, render_template, request, redirect, session
from pymongo import MongoClient
import joblib
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'secret123'  # Use a secure key in production

# Load model & vectorizer
model = joblib.load('model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# MongoDB connection
client = MongoClient('mongodb://localhost:27017/')
db = client['job_data']
users = db['users']
predictions = db['predictions']

# Routes
@app.route('/')
def index():
    if 'user' in session:
        return render_template('home.html')
    return redirect('/login')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        if users.find_one({'username': uname}):
            return "User exists!"
        users.insert_one({'username': uname, 'password': generate_password_hash(pwd)})
        return redirect('/login')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        user = users.find_one({'username': uname})
        if user and check_password_hash(user['password'], pwd):
            session['user'] = uname
            return redirect('/')
        return "Invalid!"
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect('/login')

@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return redirect('/login')
    desc = request.form['description']
    vec = vectorizer.transform([desc])
    pred = model.predict(vec)[0]
    result = "Fraudulent" if pred == 1 else "Legitimate"
    predictions.insert_one({'username': session['user'], 'description': desc, 'prediction': result})
    return render_template('home.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)


```


```python

<!DOCTYPE html>
<html>
<head><title>Login</title></head>
<body style="text-align:center;">
    <h2>Login</h2>
    <form method="POST">
        <input name="username" placeholder="Username" required><br><br>
        <input name="password" type="password" placeholder="Password" required><br><br>
        <button type="submit">Login</button>
    </form>
    <p>No account? <a href="/register">Register</a></p>
</body>
</html>


```


```python


<!DOCTYPE html>
<html>
<head><title>Register</title></head>
<body style="text-align:center;">
    <h2>Register</h2>
    <form method="POST">
        <input name="username" placeholder="Username" required><br><br>
        <input name="password" type="password" placeholder="Password" required><br><br>
        <button type="submit">Register</button>
    </form>
    <p>Have an account? <a href="/login">Login</a></p>
</body>
</html>

```

```python

<!DOCTYPE html>
<html>
<head><title>Fraud Detector</title></head>
<body style="text-align:center;">
    <h2>Welcome {{ session['user'] }}</h2>
    <a href="/logout">Logout</a><br><br>

    <form method="POST" action="/predict">
        <textarea name="description" rows="8" cols="60" placeholder="Enter job description..." required></textarea><br><br>
        <button type="submit">Check Fraud</button>
    </form>

    {% if prediction %}
        <h3>Prediction: {{ prediction }}</h3>
    {% endif %}
</body>
</html>


```
