from os.path import join
import numpy as np
from scipy.misc import imread, imsave, imresize
from keras.models import load_model

input_size = (128, 128)
label_size = (256, 256)
saved_model_dir = 'BSDS500Resized/model'
image_dir = 'a10.jpg'

#%%
image = imread(image_dir, flatten=False, mode='L')
scaled = imresize(image, input_size, 'bicubic')
scaled = imresize(scaled, label_size, 'bicubic')

image =imresize(image, label_size, 'bicubic')
#%% Load model and weights
model = load_model(join(saved_model_dir, 'model.h5'))

#%% Reshape scaled image for input to model
input_img = np.expand_dims(scaled, axis=0)
input_img = np.transpose(input_img, (1, 2, 0))

#%%
input_img = np.expand_dims(input_img, axis=0)
#%%
output = model.predict(input_img)

#%%
output = np.squeeze(output)

#%%
imsave("scaled_image.jpg", scaled)
imsave("output_image.jpg", output)
imsave("reference_image.jpg", image)
