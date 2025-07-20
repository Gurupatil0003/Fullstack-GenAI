```python

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SPAM Detector</title>
</head>
<body>
    <h1>SPAM Detector</h1>

    <form action="/" method="post">
        <textarea name="message" rows="4" cols="50" placeholder="Enter a message"></textarea><br><br>
        <button type="submit">Check</button>
    </form>

    {% if message %}
        <h2>Message: {{ message }}</h2>
        <h3>{{ result }}</h3>
    {% endif %}
</body>
</html>



```

```
from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np 
import joblib

app = Flask(__name__)

model = joblib.load('spam_model.pkl')

@app.route('/',methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    message = request.form.get('message')
    output = model.predict([message])
    if output == [0]:
      result = "This Message is Not a SPAM Message."
    else:
      result = "This Message is a SPAM Message." 
    return render_template('index.html', result=result,message=message)      

  else:
    return render_template('index.html')  


if __name__ == '__main__':
    app.run(debug=True)


```

```python
from flask import Flask, jsonify, request

app = Flask(__name__)

books = [
    {"id": 1, "title": "To Kill a Mockingbird"},
    {"id": 2, "title": "1984"},
    {"id": 3, "title": "Pride and Prejudice"}
]

@app.route('/books', methods=['GET', 'POST'])
def books_list():
    if request.method == 'GET':
        return jsonify(books)
    data = request.json
    new_book = {"id": len(books) + 1, "title": data['title']}
    books.append(new_book)
    return jsonify(new_book), 201

@app.route('/books/<int:book_id>', methods=['GET', 'PUT', 'DELETE'])
def single_book(book_id):
    book = next((b for b in books if b['id'] == book_id), None)
    if not book:
        return jsonify({"error": "Book not found"}), 404

    if request.method == 'GET':
        return jsonify(book)
    elif request.method == 'PUT':
        book['title'] = request.json['title']
        return jsonify(book)
    elif request.method == 'DELETE':
        books.remove(book)
        return '', 204

if __name__ == '__main__':
    app.run(debug=True)



```
