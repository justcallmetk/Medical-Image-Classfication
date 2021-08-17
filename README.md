# Abstract


I began this Project in a hope to further the applications of Machine Learning(ML) in medical imaging
while using blockchain technology to ensure security and the integrity of the results.

The convolutional neural network, also known as convert nets
or CNN's, is commonly used to identify patterns,
distinguish images, detect objects and recognize faces.


This current implementation of a CNN trains a tensorflow based CNN to detect
abnormalities in chest x rays using a subset of the NIH chest X rays (C.X.R.). 
This ML model in its final form will be able to process 
medical images (xrays, ctscans , ekg etc.), train, learn, identify various images from different reigions 
of the body and dectect abnormalities.

# Medical-Image-Classfication

In this project, you'll be using a subset of the NIH chest X
rays or C. X. as our data set for training.

For Now We will be using only two of the potential classes, effusion
and no finding (no anomaly was detected).

Effusion is a medical condition where liquid gets collected
in lungs due to various diseases such as tuberculosis,
pneumonia or even heart failure.


# Making Predictions in Inference 

Using our ResnetBuilder in resnet.py
Best model weights that we developed and saved during our
training iterations  will be loaded into this to do that.

# Tips
Remember to Use Correct File Path in order to use NIH C.X.R. data 
as well as import data to the Image Classification Inference file. 



