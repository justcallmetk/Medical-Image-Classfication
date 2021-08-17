# Abstract


I began this image classification project in hope to further the applications of Machine Learning(ML) in medical imaging
while using blockchain technology to ensure security and the integrity of the results.

The convolutional neural network, also known as CNN's, is commonly used to identify patterns,
distinguish images, detect objects and recognize faces.

This current implementation of a CNN trains a tensorflow based CNN to detect
abnormalities in chest x rays using a subset of the National Institute of Health's chest x-rays (C.X.R.)
CXR Dataset: (https://www.kaggle.com/nih-chest-xrays/data )

This ML model in its final form will be able to process 
medical images (x-rays, ctscans , ekg etc.), train, learn, identify various images from different regions 
of the body and detect abnormalities.

# Medical-Image-Classfication

In this project, you'll be using a subset of the National Institute of Health 
chest X-rays (C.X.R.) as our data set for training. 


For now we will be using only two of the potential classes, effusion
and no finding (no anomaly was detected).

Effusion is a medical condition where liquid gets collected
in lungs due to various diseases such as tuberculosis,
pneumonia and heart failure.

# Blockchain Implementation

At this time, I am working on the Blockchain implementation via test networks.


# Making Predictions in Inference 

Using our ResnetBuilder in resnet.py
Best model weights that we developed and saved during our
training iterations  will be loaded into this to do that.

# Tips

Remember to use Correct File Path in order to use NIH C.X.R. data 
as well as import data to the Image Classification Inference file. 

Refer to the requirements.txt file to ensure that you meet the requirements to
implement scripts.


National Institute of Health Chest X-ray Dataset: https://www.kaggle.com/nih-chest-xrays/data
