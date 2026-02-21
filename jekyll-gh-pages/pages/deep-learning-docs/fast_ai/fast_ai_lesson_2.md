---
title: Fast AI Lesson 2
tags: 
keywords: Fast AI Lesson 2
last_updated: February 20, 2026
summary: 
sidebar: mydoc_sidebar_haza
permalink: fast_ai_lesson_2.html
folder: deep-learning-docs/fast_ai
---

## Lesson 2

* [https://course.fast.ai/Lessons/lesson2.html](https://course.fast.ai/Lessons/lesson2.html)
* [https://www.youtube.com/watch?v=F4tvM4Vb3A0](https://www.youtube.com/watch?v=F4tvM4Vb3A0)
* [https://github.com/hmojicag/deep-learning/tree/main/fast-ai/fastbook/02_production.ipynb](https://github.com/hmojicag/deep-learning/tree/main/fast-ai/fastbook/02_production.ipynb)

## Issues with downloading images for bears

Running the jupyter notebook for the session 2 just as is didn't work when it comes to downloading the images for bears.

The notebook is written to download images from **Bing**, but **Bing** is no longer supporting an API, **Bing API** is no more.

Well, I tried to adapt the code to download from **Duck Duck Go** as with the previous lesson, but it seems that Duck Duck Go is also blocking the requests.

The error was an `HTTPError: HTTP Error 403: Forbidden`.

So, I ended up just going to Duck Duck Go Images, scroll down until you have good enough images or DDG is no longer returning results and then I used a Chrome Extension to download all the images from that search result.

I mean, is no longer "downloading" at that point because you are just saving the images that are already in the browser cache, but you get the point, the extension name is `Download All Images`.

The image files didn't get saved with an extension, so I had to rename them all to `*.jpg`.

I put all the images in the folder [bears](https://github.com/hmojicag/deep-learning/tree/main/fast-ai/fastbook/images/bears)

And then I used that in the notebook, skipping the part where it tries to download the images.

## Training, exporting and using the model

I wanted some hands-on experience with training a model, exporting it and then using it in a different notebook.

So I created the next python scripts to do just that:

### Train and export the model

Run the script [https://github.com/hmojicag/deep-learning/blob/main/fast-ai/lesson2/1_train_bear_detector_model.py](https://github.com/hmojicag/deep-learning/blob/main/fast-ai/lesson2/1_train_bear_detector_model.py)

It will train the model using the data in the folder.

```sh
python fast-ai/lesson2/1_train_bear_detector_model.py
```

Sample output
```txt
Training model with data from fast-ai/fastbook/images/bears and saving to fast-ai/lesson2
epoch     train_loss  valid_loss  error_rate  time    
0         1.049577    0.328317    0.070312    00:06                                                                                                     
epoch     train_loss  valid_loss  error_rate  time    
0         0.361420    0.241619    0.050781    00:08                                                                                                     
1         0.240842    0.277644    0.062500    00:08                                                                                                     
2         0.184921    0.221268    0.042969    00:08                                                                                                     
3         0.142347    0.205505    0.050781    00:08                                                                                                     
Saving model to fast-ai/lesson2
```

### Use the model to make inferences on bear images

Run the script [https://github.com/hmojicag/deep-learning/blob/main/fast-ai/lesson2/2_load_model_and_infer.py](https://github.com/hmojicag/deep-learning/blob/main/fast-ai/lesson2/2_load_model_and_infer.py)

It will load the model just trained and it will make inferences on the images on the folder `fast-ai/lesson2/bear-samples` and print the results to the console.

```sh
python fast-ai/lesson2/2_load_model_and_infer.py
```

Sample output
```txt
Model loaded successfully
Running inference on images from fast-ai/lesson2/bear-samples
Image: black-bear-1.jpg, Prediction: black, Probability: 1.0000                                                                  
Image: black-bear-2.jpg, Prediction: black, Probability: 0.9999                                                                  
Image: grizly-bear-1.jpeg, Prediction: grizzly, Probability: 1.0000                                                              
Image: black-bear-3.jpeg, Prediction: black, Probability: 0.9998                                                                 
Image: grizly-bear-3.jpg, Prediction: grizzly, Probability: 1.0000                                                               
Image: teddy-bear-1.jpg, Prediction: teddy, Probability: 1.0000                                                                  
Image: teddy-bear-2.jpg, Prediction: teddy, Probability: 1.0000                                                                  
Image: teddy-bear-3.jpg, Prediction: teddy, Probability: 1.0000                                                                  
Image: girzly-bear-2.jpg, Prediction: grizzly, Probability: 0.9922   
```

## Deploy to production

Now I want to use this model I just trained and make it available in a production environment.

With that I will be able to be able to demonstrate the full cycle.

This is still using the fastai library but still is a step forward in the right direction.

### My first Flask

Just to test Flask I created a simple script that seems to be spinning up a web server and responding to requests.
[3_flask_website_test.py](../../../../fast-ai/lesson2/3_flask_website_test.py)

This is interesting, Python has this very minimal approach compared to other languages and frameworks.

### Building a more complex site

I'm following this documentation: [https://flask.palletsprojects.com/en/stable/quickstart/#a-minimal-application](https://flask.palletsprojects.com/en/stable/quickstart/#a-minimal-application)

Run the Flask application located here:

[4_flask_website](../../../../fast-ai/lesson2/4_flask_website)

