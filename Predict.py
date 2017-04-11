import numpy as np
from scipy.misc import imread, imsave, imresize
from keras.models import model_from_json
from os.path import join

#%% Define variables
input_size = (128, 128)
label_size = (256, 256)
saved_model_dir = 'BSDS500Resized/model/model.json'
saved_weights_dir = 'BSDS500Resized/model/model_weights.h5'
output_dir = 'output'

image_dir = 'a10.jpg'


#%% Rescale images 
image = imread(image_dir, flatten=False)
scaled = imresize(image, input_size, 'bicubic')
scaled = imresize(scaled, label_size, 'bicubic')
image =imresize(image, label_size, 'bicubic')


#%% Load model and weights separately due to error in keras
model = model_from_json(open(saved_model_dir).read())
model.load_weights(saved_weights_dir)


#%% Reshape scaled image for input to model (must be 4 dimensions with channels last)
#input_img = np.expand_dims(scaled, axis=0)
#input_img = np.transpose(scaled, (1, 2, 0))
input_img = np.expand_dims(scaled, axis=0)


#%%
output = model.predict(input_img)


#%% Format output to 2-D tensor for saving image
output = np.squeeze(output)


#%% Save images
imsave(join(output_dir, "scaled_image.jpg"), scaled)
imsave(join(output_dir, "output_image.jpg"), output)
imsave(join(output_dir, "reference_image.jpg"), image)
