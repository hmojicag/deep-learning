from fastai.text.all import *

if __name__ == "__main__":
    path = untar_data(URLs.IMDB)
    dls = TextDataLoaders.from_folder(path, valid='test', bs=16, device="cuda")
    learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
    learn.fine_tune(4, 1e-2)

    learn.predict("I really liked that movie!")
    learn.predict("This movie truly sucks")
    learn.predict("The director of that movie didn't know what it was doing")
    learn.predict("I could watch it everyday all day")

# (.venv) PS C:\Users\final\repos\deeplearning> python .\fast-ai\lesson1\imdb-text-classifier-cuda.py
# epoch     train_loss  valid_loss  accuracy  time
# 0         0.497177    0.404711    0.816720  09:27
# epoch     train_loss  valid_loss  accuracy  time
# 0         0.341843    0.265538    0.896040  51:34
# 1         0.245558    0.235724    0.906640  51:41
# 2         0.210026    0.192282    0.927080  51:39
# 3         0.152941    0.201258    0.925320  51:44