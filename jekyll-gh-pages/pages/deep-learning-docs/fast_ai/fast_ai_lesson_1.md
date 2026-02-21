---
title: Fast AI Lesson 1
tags: 
keywords: Fast AI, Lesson 1
last_updated: February 20, 2026
summary:
sidebar: mydoc_sidebar_haza
permalink: fast_ai_lesson_1.html
folder: deep-learning-docs/fast_ai
---

## Lesson 1
* [https://course.fast.ai/Lessons/lesson1.html](https://course.fast.ai/Lessons/lesson1.html)
* [https://www.youtube.com/watch?v=8SF_h3xF3cE](https://www.youtube.com/watch?v=8SF_h3xF3cE)

Go through
1. (https://www.kaggle.com/code/jhoward/jupyter-notebook-101)[https://www.kaggle.com/code/jhoward/jupyter-notebook-101]
2. [https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data](https://www.kaggle.com/code/jhoward/is-it-a-bird-creating-a-model-from-your-own-data)

## Python stuff

How to install and manage Python in Mac: [https://eddieantonio.ca/blog/2020/01/26/installing-python-on-macos/](https://eddieantonio.ca/blog/2020/01/26/installing-python-on-macos/)

1. Install Homebrew
2. Instead of install `python` directly, use `pyenv`
	1. `brew install pyenv`
3. Modify PATH to add `shims` for `pyenv`
	1. Edit `~/.zshrc`
	2. `export PATH="$HOME/.pyenv/shims:$PATH"`
4. Check python versions installed
	1. `pyenv versions`
5. List python versions available to install
	1. `pyenv install -l`
6. Install python version
	1. `pyenv install 3.14.2`
7. Choose your default Python version
	1. `pyenv global 3.14.2`

I made everything work with `Python version 3.13.0`.

## Python virtual environments

```sh
# Create virtual environment
python -m venv .venv

# Use virtual environment
source .venv/bin/activate
.\.venv\Scripts\Activate.ps1

# Exit virtual environment
deactivate

# Destroy virtual environment
rm -rf .venv

```

## Jupyter Notebooks

```sh
# Install Jupyter Notebooks globally
pip install notebook ipykernel ipython fastai flask

# Create a virtual environment inside the Jupyter Notebooks folder (fastbook)
# Register Environment as Jupyter Kernel
python -m ipykernel install --user --name=fastbook-pyenv --display-name "Fastbook PyEnv"

# Start the jupyter server in this directory
jupyter notebook
```


## Error. Segmentation fault running fastai

A segmentation fault (SIGSEGV) in Python 3.14 on macOS, particularly with heavy libraries like fastai, is likely caused by binary incompatibility with pre-release or new Python versions, or memory management bugs.
Given Python 3.14 is very new, it is likely that `fastai`, `torch`, or `numpy` dependencies are not yet stable, causing C-extension crashes.

1. **Downgrade Python:** Use `pyenv` to switch to Python 3.11 or 3.12, as 3.14 is in early development/alpha stages.
2. **Reinstall Libraries:** Reinstall `fastai` and `torch` to ensure they are compiled for your current architecture (Intel or Apple Silicon).
3. **Update Dependencies:** Ensure `numpy` and `fastai` are fully updated, as newer versions may address compatibility issues.
**Potential Causes & Solutions**

- **Incompatible C-Extensions:** Segmentation faults in Python are rare unless a C-extension (like `torch` or `numpy`) causes a memory error. The new Python 3.14 garbage collector or interpreter might be breaking older compiled libraries.
- **Mac Architecture (M1/M2/M3):** If on Apple Silicon, ensure you are not running `x86_64` Python via Rosetta, as this frequently causes segfaults. Run `python --version` and check that it is `arm64`: `python -c "import platform; print(platform.machine())"`

**Solucion:** 
I was able to make it all work just fine with **Python version 3.10.0**

## Error Jupyter ProgressCallback NBMasterBar

Error ejecutando `learn.fine_tune(1)` directo en Jupyter Notebooks.

AttributeError: Exception occured in `ProgressCallback` when calling event `before_fit`: 'NBMasterBar' object has no attribute 'out' in jupyter

**Solucion:**

**Uninstall the current `fastprogress` package** by running the following command
```sh
pip uninstall fastprogress -y
```

**Install the specific compatible version (1.0.3)** using the following command
```sh
pip install "fastprogress==1.0.3"
```

## Error RuntimeError: MPS backend out of memory

RuntimeError: MPS backend out of memory (MPS allocated: 13.23 GiB, other allocations: 6.92 GiB, max allowed: 20.13 GiB). Tried to allocate 20.25 MiB on private pool. Use PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable upper limit for memory allocations (may cause system failure).

**Solution**

Reduce the batch size, it is usually set to 64 by default, using 16 fixed the issue, the memory used was just too much.

Example:
```python
dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test', bs=16, device="mps")
```

Passing `bs=16` seem to have fixed the issue.
## Error. MPSNDArray.mm MPSNDArrayDescriptor sliceDimension:withSubrange:

AppleInternal/Library/BuildRoots/4~B_wcugCOyFEmrl3129h8l5wJX874wFxy1jG_pok/Library/Caches/com.apple.xbs/Sources/MetalPerformanceShaders/MPSCore/Types/MPSNDArray.mm:124: failed assertion `[MPSNDArrayDescriptor sliceDimension:withSubrange:] error: subRange.start (399) is not less than length of dimension[2] (1)'

The issue seem to be because of Apple MPS (Metal Performance Shaders): https://developer.apple.com/metal/pytorch/

https://docs.pytorch.org/docs/stable/notes/mps.html

I was not able to fix  the issue but not using `MPS` and forcing `cpu` in `fastai` and `pytorch` with `device="cpu"` made the issue not appear again, although training the model is way slower.

Example:
```python
dls = TextDataLoaders.from_folder(untar_data(URLs.IMDB), valid='test', bs=256, device="cpu")
```

But like, super slow, it took more than 2hrs to complete:
```
(.venv) hmojica@HazaMacBookPro deeplearning % python fast-ai/lesson1/imdb-text-classifier.py
epoch     train_loss  valid_loss  accuracy  time    
0         0.470989    0.399647    0.818320  11:36                                                                                                       
epoch     train_loss  valid_loss  accuracy  time    
0         0.347313    0.300254    0.875400  30:52                                                                                                       
1         0.278736    0.217268    0.914320  27:12                                                                                                       
2         0.229203    0.204738    0.919760  27:05                                                                                                       
3         0.200575    0.202075    0.922920  27:22
```


## Error. FileNotFoundError No such file or directory ounter.pkl'

FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\final\\.fastai\\data\\imdb_tok\\counter.pkl'

https://necromuralist.github.io/Neurotic-Networking/index.html
https://github.com/fastai/fastai/issues/2787
https://github.com/fastai/fastai/issues/2787#issuecomment-1585705311

Using `if __name__ == "__main__":` seems to have fixed the issue? mamadas...
```python
if __name__ == "__main__":  
    path = untar_data(URLs.IMDB)  
    dls = TextDataLoaders.from_folder(path, valid='test', bs=16, device="cuda")  
    learn = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)  
    learn.fine_tune(4, 1e-2)
```