import os
from fastai.text.all import *

# Batch Size (bs) parameter is 64 by default if not passed.
# check fastai/data/core.py
# Reduce the batch size if memory issues happen.
# The device parameter is set to "cpu" to ensure that the model runs on the CPU,
# which is necessary because MPS did not work well in my MacBook.
dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test', bs=256, device="cpu")
learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)
learn.fine_tune(4, 1e-2)