
```python
from pymongo import MongoClient
import gridfs

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client['hotel_booking']
fs = gridfs.GridFS(db)

# Hotel data
hotels_data = [
    {"name": "Taj Hotel", "location": "Mumbai", "price": 5000, "image": "taj.jpg"},
    {"name": "Oberoi Hotel", "location": "Delhi", "price": 4500, "image": "oberoi.jpg"},
    {"name": "Leela Palace", "location": "Bangalore", "price": 6000, "image": "leela.jpg"}
]

# Insert hotels with images
for hotel in hotels_data:
    with open(f"static/images/{hotel['image']}", "rb") as img_file:
        image_id = fs.put(img_file, filename=hotel['image'])
        db.hotels.insert_one({ 
            "name": hotel['name'],
            "location": hotel['location'],
            "price": hotel['price'],
            "image_id": image_id
        })

print("Hotels with images inserted successfully!")


```

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
