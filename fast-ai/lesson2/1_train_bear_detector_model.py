# Execute this file from the root of the repository
# python fast-ai/lesson2/1_train_bear_detector_model.py

from fastcore.all import *
from fastai.vision.all import *

def train_bear_detector_model():
    path = Path('fast-ai/fastbook/images/bears')
    outputPath = Path('fast-ai/lesson2/')

    print(f"Training model with data from {path} and saving to {outputPath}")
    bears = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=get_image_files,
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        get_y=parent_label,
        item_tfms=Resize(128))

    bears = bears.new(
        item_tfms=RandomResizedCrop(224, min_scale=0.5),
        batch_tfms=aug_transforms())

    dls = bears.dataloaders(path)

    learn = vision_learner(dls, resnet18, metrics=error_rate, path=outputPath)
    learn.fine_tune(4)

    print(f"Saving model to {outputPath}")
    learn.export()

if __name__ == "__main__":
    train_bear_detector_model()
