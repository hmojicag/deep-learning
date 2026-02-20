from fastai.text.all import *

if __name__ == "__main__":
    path = untar_data(URLs.IMDB)
    dls = TextDataLoaders.from_folder(path, valid='test', bs=64, device="cuda")
    learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
    learn.fine_tune(4, 1e-2)

    print(learn.predict("I really liked that movie!"))
    print(learn.predict("This movie truly sucks"))
    print(learn.predict("The director of that movie didn't know what it was doing"))
    print(learn.predict("I could watch it everyday all day"))

# (.venv) PS C:\Users\final\repos\deeplearning> python .\fast-ai\lesson1\imdb-text-classifier-cuda-64.py
# epoch     train_loss  valid_loss  accuracy  time
# 0         0.462926    0.397392    0.819920  05:03
# epoch     train_loss  valid_loss  accuracy  time
# 0         0.314941    0.254108    0.893440  54:22
# 1         0.243437    0.220717    0.909240  54:22
# 2         0.210634    0.182527    0.930440  54:19
# 3         0.166536    0.186173    0.931000  54:39
# ('pos', tensor(1), tensor([0.0012, 0.9988]))
# ('neg', tensor(0), tensor([0.9823, 0.0177]))
# ('neg', tensor(0), tensor([0.5386, 0.4614]))
# ('pos', tensor(1), tensor([0.0088, 0.9912]))