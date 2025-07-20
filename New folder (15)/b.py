# app.py
from flask import Flask, request, render_template_string
import mlflow.sklearn

# Replace with your run ID (from train.py output)
RUN_ID = "03efca32bb1042c1a30dbdb25b68cb19"
MODEL_URI = f"runs:/{RUN_ID}/iris_model"
model = mlflow.sklearn.load_model(MODEL_URI)
from flask_ngrok import run_with_ngrok

app = Flask(__name__)


run_with_ngrok(app)  # Start ngrok when app.run() is called


HTML_TEMPLATE = """
<!doctype html>
<title>Iris Predictor</title>
<h2>Iris Flower Predictor</h2>
<form action="/predict" method="post">
  Sepal Length: <input type="text" name="sl"><br>
  Sepal Width: <input type="text" name="sw"><br>
  Petal Length: <input type="text" name="pl"><br>
  Petal Width: <input type="text" name="pw"><br>
  <input type="submit" value="Predict">
</form>
{% if prediction %}
  <h3>Prediction: {{ prediction }}</h3>
{% endif %}
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    sl = float(request.form["sl"])
    sw = float(request.form["sw"])
    pl = float(request.form["pl"])
    pw = float(request.form["pw"])
    prediction = model.predict([[sl, sw, pl, pw]])[0]
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run()
