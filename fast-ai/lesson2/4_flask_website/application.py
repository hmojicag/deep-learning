# pip install flask
# flask --app fast-ai/lesson2/4_flask_website/application.py run --host=0.0.0.0 --port=8000
# Open a browser and navigate to http://localhost:8000
import os
from flask import request
from flask import Flask, render_template, redirect, url_for, send_from_directory, send_file
from werkzeug.utils import secure_filename
from fastcore.all import *
from fastai.vision.all import *

UPLOAD_FOLDER = 'fast-ai/lesson2/4_flask_website/uploads'
DOWNLOAD_FOLDER = 'uploads' # Same as UPLOAD_FOLDER but relative to this file location
BEAR_MODEL_LOCATION = 'fast-ai/lesson2/export.pkl'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['BEAR_MODEL_LOCATION'] = BEAR_MODEL_LOCATION

class IndexContext:
    def __init__(self, errorMsg):
        self.errorMsg = errorMsg
        self.imgUrl = ""
        self.infoMsg = ""

def load_bear_detector_model():
    # Path to the export from the previous file
    path = Path(app.config['BEAR_MODEL_LOCATION'])
    print(f"Loading model from {path}")
    learn_inf = load_learner(path)
    print("Model loaded successfully")
    return learn_inf

def infer_bear_to_single_image(learn_inf, image_path):
    print(f"Running inference on image {image_path}")
    img = PILImage.create(image_path)
    pred, pred_idx, probs = learn_inf.predict(img)
    print(f"Image: {image_path}, Prediction: {pred}, Probability: {probs[pred_idx]:0.4f}")
    return pred, pred_idx, probs

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=['GET'])
def index():
    indexContext = IndexContext("")
    return render_template("index.html", context=indexContext)

@app.route('/', methods=['POST'])
def upload_file():
    indexContext = IndexContext("")
    if 'bearImageFile' not in request.files:
        indexContext.errorMsg = 'No file part'
        return render_template("index.html", context=indexContext)
    file = request.files['bearImageFile']
    # If the user does not select a file, the browser submits an
    # empty file without a filename.
    if file.filename == '':
        indexContext.errorMsg = 'No selected file'
        return render_template("index.html", context=indexContext)
    if file and allowed_file(file.filename):
        indexContext.imgUrl = '/uploads/' + file.filename
        filename = secure_filename(file.filename)
        img_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(img_file_path)
        learn_inf = load_bear_detector_model()
        pred, pred_idx, probs = infer_bear_to_single_image(learn_inf, img_file_path)
        indexContext.infoMsg = f"Prediction: {pred}, Probability: {probs[pred_idx]:0.4f}"
        return render_template("index.html", context=indexContext)
    return redirect(url_for('index'))

@app.route('/uploads/<name>')
def download_file(name):
    print(name)
    print(app.config['UPLOAD_FOLDER'])
    print(os.path.join(app.config['UPLOAD_FOLDER']))
    print(os.path.join(app.config['UPLOAD_FOLDER'], name))
    # For some weird reason, send_from_directory doesn't work with the UPLOAD_FOLDER,
    # because it is already concatenating the full os path up to the location of this file
    return send_from_directory(app.config["DOWNLOAD_FOLDER"], name)

@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name=None):
    return render_template('hello.html', person=name)