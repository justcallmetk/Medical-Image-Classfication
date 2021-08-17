import tensorflow              # building network
from skimage import io    # rescale and preprocess
import os                              #  directory paths for images
import glob                          #  directory paths for images
import numpy as np         # array and matrix manipulation 
import matplotlib.pyplot as plt   #  Visualize Data
from tensorflow.python.keras.utils.data_utils import Sequence

print(“Imports Completed”)

import warnings 
warnings.simplefilter(‘ignore’)

#Load Data Set
DATASET_PATH = ‘C:\Data\CXR_data_UpGrad’
print(DATASET_PATH)
file path = os.path.join(DATASET_PATH, ‘models\\best_model.hdf5’)
print(filepath)


# Classes of Images
disease_cls = [‘effusion’, ‘nofinding’] 

# Read the Effusion and No finding images
effusion_path = os.path.join(DATASET_PATH, disease_cls[0], ‘*’)  # ‘*’ includes every image
effusion = glob.glob(effusion_path)
effusion = io.imread(effusion[0])


normal_path = os.path.join(DATASET_PATH, disease_cls[1], ‘*’)
normal = glob.glob(normal_path)
normal = io.imread(normal[0])

f, axes = put.subplots(1,2,sharey= True) # CXR data only goes up and down… Do not center crop image
f.set_figwidth(10)


axes[0].imshow(effusion, cmap=‘gray’)
axes[1].imshow(normal, cmap=‘gray’)

#Create Data Generator

from skimage.transform import rescale
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
	featurewise_center = True,
	featurewise_std_normalization = True,
	rotation_range =10,
	width_shift_range = 0,
	height_shift_range = 0,
	vertical_flip = False)



 # Preprocess_img Class Creation

def preprocess_img( img, mode ):
	img = ( img - img.min() / (img.max() - img.min() )
	img = rescale(omg, 0.25, multichannel = True, mode = ‘constant’)

	if mode == ‘train’:
		if np.random.randn() > 0:
			img = datagen.random_transform(img)
	return img



# Residual Convolutional Neural Network Layers
# Deep Learning Technique

import resent

img_channels =1
img_rows = 256
img_cols = 256

nb_classes =2


# Augmented Data Generator Class

class AugmentedDataGenerator(eras.utils.Sequence):
	def _inti_(self, mode = ‘train’, ablation = None, disease_cls = [‘effusion’, ‘nofinding’])
			batch_size = 32, dim (256, 256), channels = 1, shuffle = True): 
		self.dim = dim
		sefl.batch_sixe = batch_sixe
		self.labels={}
		self.list_IDs = []
		self.mode = mode

		for i, cls in enumerate(disease_cls):
			paths = glob.glob(os.path.join(DATASET_PATH, cls, “*”))
			brk_point = int(len(paths)* 0.8) #Train using 80% of data
			
			if self.mode == ‘train’:
				paths = paths[:brk_point]
			else:
				paths = paths[brk_point:]
			if ablation is not None:
				paths = paths[:int(len(paths)*ablation/100]]
			self.list_IDs+=paths
			self.labels.update({p:i for p in paths})
			self.n_channles = n_channels
			self.n_clases = len(disease_cls)
			self.shuffle = shuffle
			self.on_epoch_end()


	def_len_(self):
		return int(np.floor(len(self.list_IDs / self.batch_size))
 	def_getitem_(self,index):
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size
		list_IDs_temp = [self.list_ids[k] for k in indexes]

		X,y = self._data_generation(list_IDs_temp)
		return X, y

	def on_epoch_end(self):
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)



	def _data_generation(self, list_IDs_temp):
		X = np.empty((self.batch_size, *self.dim,n_channels))
		y  np.empty((self.batch_seize), dtype=int)

		delete_rows = []

		for i, ID in enumerate(list_IDs_temp)
			img = io.imread(ID)
			img[ : , : , np.newaxis ]
			if img.shape == (1024, 1024, 1):
				img = preprocess_img( img, self.mode )
				X[i,] = img
				y[i] = self.labels[ID]
			else:
				delete_rows.append(i)
				continue

		X = np.delete(X, delete_rows, axis=0)
		
		y = np.delete(y, delete_rows, axis=0)
		return X, tensorflow.keras.utils.to_catergorical(y, num_classes = self.n_classes)
		

# Resnet Model Building, Compiling and Model Fitting

model = resent.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)

model.compile(loss='categorical_crossentropy',optimizer='SGD', metrics=['accuracy'])
training_generator = AugmentedDataGenerator('train',ablation = 5)
validation_generator = AugmentedDataGenerator('val',ablation = 5)
model.fit(training_generator,validation_data=None, epochs=5)

from sklearn.metrics import roc_auc_score
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import *


class roc_callback(Callback):

    def on_train_begin(self,logs={}):
        logs['val_auc'] = 0

    def on_epoch_end(self,epoch,logs={}):
        y_p=[]
        y_v=[]
        for i in range(len(validation_generator)):
            x_val,y_val = validation_generator[i]
            y_pred=self.model.predict(x_val)
            y_p.append(y_pred)
            y_v.append(y_val)
        y_p=np.concatenate(y_p)
        y_v=np.concatenate(y_v)
        try:
            roc_auc = roc_auc_score(y_v,y_p, average = 'micro')
        except ValueError:
            pass
        print('\nVal AUC for epoch{}:{}'.format(epoch,roc_auc))
        logs['val_auc'] = roc_auc


model = resnet.ResnetBuilder.build_resnet_18((img_channels,
                                              img_rows,img_cols), nb_classes)
model.compile(loss='categorical_crossentropy',optimizer='SGD', metrics=['accuracy'])
training_generator = AugmentedDataGenerator('train',ablation = 20)
validation_generator = AugmentedDataGenerator('val',ablation = 20)

auc_logger = roc_callback()

history = model.fit(training_generator,validation_data=validation_generator, epochs=5, callbacks=[auc_logger])

from functools import partial
import tensorflow.keras.backend as K
from itertools import product



def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    cross_ent = K.categorical_crossentropy(y_true, y_pred, from_logits=False)
    return cross_ent * final_mask

bin_weights = np.ones((2,2))
bin_weights[0, 1] = 5
bin_weights[1, 0] = 5
ncce = partial(w_categorical_crossentropy, weights=bin_weights)
ncce.__name__ ='w_categorical_crossentropy'
model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
model.compile(loss=ncce, optimizer='SGD',
              metrics=['accuracy'])

training_generator = AugmentedDataGenerator('train', ablation=5)
validation_generator = AugmentedDataGenerator('val', ablation=5)

model.fit(training_generator,
                    validation_data=None,
                    epochs=1)



# Decaying Learning Rate
class DecayLR(tensorflow.keras.callbacks.Callback):
    def __init__(self, base_lr=0.01, decay_epoch=1):
        super(DecayLR, self).__init__()
        self.base_lr = base_lr
        self.decay_epoch = decay_epoch
        self.lr_history = []

    def on_train_begin(self, logs={}):
        K.set_value(self.model.optimizer.lr, self.base_lr)

    def on_epoch_end(self, epoch, logs={}):
        new_lr = self.base_lr * (0.5**(epoch//self.decay_epoch))
        self.lr_history.append(K.get_value(self.model.optimizer.lr))
        K.set_value(self.model.optimizer.lr, new_lr)
model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
bin_weights = np.ones((2,2))
bin_weights[0, 1] = 10
bin_weights[1, 0] = 10
ncce = partial(w_categorical_crossentropy, weights=bin_weights)
ncce.__name__ ='w_categorical_crossentropy'
model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
model.compile(loss=ncce, optimizer='SGD',
              metrics=['accuracy'])

training_generator = AugmentedDataGenerator('train', ablation=50)
validation_generator = AugmentedDataGenerator('val', ablation=50)


#Join DATASET path to 'models\\best_model.hdf5

filepath = os.path.join(DATASET_PATH, 'models\\best_model.hdf5') 
checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(filepath, monitor= 'val_accuracy', verbose=1, save_best_only=True,mode='max') #Training Checkpoint

decay = DecayLR()

history = model.fit(training_generator,validation_data=validation_generator, epochs=10, callbacks=[auc_logger, decay, checkpoint])

val_model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
val_model.load_weights(filepath)


# To have new file for inference reintroduce effusion

effusion_path = os.path.join(DATASET_PATH, cls[0], ‘*’)
effusion = glob.glob(efffusion_path)
effusion = io.imread(effusion[0])

effusion = io.imread(effusion[-5])
val_model.compile(loss=ncce, optimizer='SGD', metrics=['val_accuracy'])


img = preprocess_img(effusion[:,:,np.newaxis,'validation'])

val_model.predict(img[np.newaxis, :]) 
