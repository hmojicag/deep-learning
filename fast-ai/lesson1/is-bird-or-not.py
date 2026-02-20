from fastcore.all import *
from fastai.vision.all import *

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_DISABLE_MPS_FALLBACK'] = '1'  # optional: avoids unexpected MPS fallbacks on Apple GPUs

import faulthandler
faulthandler.enable()

path = Path('bird_or_not')
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path, bs=2)

#dls.show_batch(max_n=6)

# Train
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

# is_bird,_,probs = learn.predict(PILImage.create('forest.jpg'))
# print(f"This is a: {is_bird}.")
# print(f"Probability it's a bird: {probs[0]:.4f}")