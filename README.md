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

# Application Requirements
absl-py==0.12.0

appdirs==1.4.4

astroid==2.5.6

astunparse==1.6.3

atomicwrites==1.4.0

attrs==21.2.0

audioread==2.1.9

cachetools==4.2.2

certifi==2020.12.5

cffi==1.14.5

chardet==4.0.0

colorama==0.4.4

coverage==5.5

cycler==0.10.0

decorator==4.4.2

ffmpeg==1.4

flatbuffers==1.12

gast==0.3.3

google-auth==1.30.0

google-auth-oauthlib==0.4.4

google-pasta==0.2.0

grpcio==1.32.0

h5py==2.10.0

idna==2.10

imageio==2.9.0

iniconfig==1.1.1

isort==5.8.0

joblib==1.0.1

Keras==2.4.3

Keras-Applications==1.0.8

Keras-Preprocessing==1.1.2

kiwisolver==1.3.1

lazy-object-proxy==1.6.0

librosa==0.8.0

llvmlite==0.36.0

Markdown==3.3.4

matplotlib==3.4.2

mccabe==0.6.1

networkx==2.5.1

numba==0.53.1

numpy==1.19.5

oauthlib==3.1.0

opt-einsum==3.3.0

packaging==20.9

pandas==1.2.4

Pillow==8.2.0

pip==21.1.1

pluggy==0.13.1

pooch==1.3.0

protobuf==3.17.0

py==1.10.0

pyarrow==4.0.0

pyasn1==0.4.8

pyasn1-modules==0.2.8

pycparser==2.20

pydub==0.25.1

pylint==2.8.2

pyparsing==2.4.7

pytest==6.2.2

pytest-cov==2.11.1

python-dateutil==2.8.1

pytz==2021.1

PyWavelets==1.1.1

PyYAML==5.4.1

requests==2.25.1

requests-oauthlib==1.3.0

resampy==0.2.2

resnet-models==1.1.3

rsa==4.7.2

scikit-image==0.18.1

scikit-learn==0.24.2

scipy==1.4.1

setuptools==56.2.0

signal-transformation==2.4.3

six==1.15.0

SoundFile==0.10.3.post1

tensorboard==2.2.2

tensorboard-data-server==0.6.1

tensorboard-plugin-wit==1.8.0

tensorflow==2.2.0

tensorflow-estimator==2.2.0

termcolor==1.1.0

threadpoolctl==2.1.0

tifffile==2021.4.8

toml==0.10.2

typing-extensions==3.7.4.3

urllib3==1.26.4

webrtcvad==2.0.10

Werkzeug==2.0.1

wheel==0.36.2

wrapt==1.12.1
