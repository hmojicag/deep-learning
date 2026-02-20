from fastai.text.all import *

if __name__ == "__main__":
    path = untar_data(URLs.IMDB)
    dls = TextDataLoaders.from_folder(path, valid='test', bs=8, device="cuda")
    learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
    learn.fine_tune(4, 1e-2)

    print(learn.predict("I really liked that movie!"))
    print(learn.predict("This movie truly sucks"))
    print(learn.predict("The director of that movie didn't know what it was doing"))
    print(learn.predict("I could watch it everyday all day"))


# (.venv) PS C:\Users\final\repos\deeplearning> python .\fast-ai\lesson1\imdb-text-classifier-cuda-8.py
# epoch     train_loss  valid_loss  accuracy  time
# 0         0.524188    0.408916    0.819000  15:07
# epoch     train_loss  valid_loss  accuracy  time
# 0         0.384456    0.308173    0.877560  47:24
# 1         0.297001    0.259738    0.911200  47:49
# 2         0.299073    0.713497    0.925560  47:46
# 3         0.169777    0.338860    0.925560  47:56
# ('pos', tensor(1), tensor([0.0775, 0.9225]))
# ('neg', tensor(0), tensor([0.7653, 0.2347]))
# ('pos', tensor(1), tensor([0.4570, 0.5430]))
# ('pos', tensor(1), tensor([0.2243, 0.7757]))