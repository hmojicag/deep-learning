# pip install flask
# flask --app fast-ai/lesson2/3_flask_website_test.py run --host=0.0.0.0
# Open a browser and navigate to http://localhost:5000
from flask import Flask

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"