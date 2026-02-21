# Execute this file from the root of the repository
# python fast-ai/lesson2/2_load_model_and_infer.py

from fastcore.all import *
from fastai.vision.all import *

def load_bear_detector_model():
    # Path to the export from the previous file
    path = Path('fast-ai/lesson2')
    print(f"Loading model from {path}")
    learn_inf = load_learner(path/'export.pkl')
    print("Model loaded successfully")
    return learn_inf

def infer_bear_to_all_images(learn_inf):
    path = Path('fast-ai/lesson2/bear-samples')
    print(f"Running inference on images from {path}")
    files = get_image_files(path)
    for file in files:
        img = PILImage.create(file)
        pred, pred_idx, probs = learn_inf.predict(img)
        print(f"Image: {file.name}, Prediction: {pred}, Probability: {probs[pred_idx]:0.4f}")

if __name__ == "__main__":
    learn_inf = load_bear_detector_model()
    infer_bear_to_all_images(learn_inf)