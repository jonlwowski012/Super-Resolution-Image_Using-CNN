import numpy as np
from scipy.misc import imread, imsave, imresize
from keras.models import model_from_json
from os.path import join
import matplotlib.pyplot as plt

#%% Define variables
input_size = (128, 128)
label_size = (256, 256)
saved_model_dir = 'image_slicing_MSE/model/model.json'
saved_weights_dir = 'image_slicing_MSE/model/model_weights.h5'
output_dir = 'image_slicing_MSE/output'
# Define the window size
windowsize_r = 32
windowsize_c = 32

image_dir = 'a10.jpg'


#%% Rescale images 
image = imread(image_dir)
scaled = imresize(image, input_size, 'bicubic')
scaled = imresize(scaled, label_size, 'bicubic')
image =imresize(image, label_size, 'bicubic')


#%% Load model and weights separately due to error in keras
model = model_from_json(open(saved_model_dir).read())
model.load_weights(saved_weights_dir)


#%% Reshape scaled image for input to model (must be 4 dimensions with channels last)
#input_img = np.expand_dims(scaled, axis=0)
#input_img = np.transpose(scaled, (1, 2, 0))
#input_img = np.expand_dims(scaled, axis=0)
input_img = scaled.copy()

#%%

input_img = np.asarray(input_img)
print(input_img.shape)

#%%
input_imgs = []
output_img = np.zeros(input_img.shape)

#%%

for r in range(0,input_img.shape[0] - windowsize_r + 1, windowsize_r):
	for c in range(0,input_img.shape[1] - windowsize_c + 1, windowsize_c):
		input_imgs.append(input_img[r:r+windowsize_r,c:c+windowsize_c])
input_imgs = np.asarray(input_imgs)

#%%
print(input_imgs.shape)
tmp = input_imgs[0,:,:,:]

#%%
output = model.predict(input_imgs)


#%%
tmp = output[0,:,:,:]

tmp2 = output[1,:,:,:]
#%%
i = 0
for r in range(0, output_img.shape[0] - windowsize_r + 1, windowsize_r):
	for c in range(0,output_img.shape[1] - windowsize_c + 1, windowsize_c):
		output_img[r:r+windowsize_r,c:c+windowsize_c] = output[i,:,:,:]
		i += 1
output_img = np.dstack((output_img[:,:,2],output_img[:,:,0],output_img[:,:,1]))
#%% Format output to 2-D tensor for saving image
#output = np.squeeze(output)

#%% Save images
imsave(join(output_dir, "scaled_image.jpg"), scaled)
imsave(join(output_dir, "output_image.jpg"), output_img)
imsave(join(output_dir, "reference_image.jpg"), image)
