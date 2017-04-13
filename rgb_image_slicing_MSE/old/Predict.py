import numpy as np
from scipy.misc import imread, imsave, imresize
from keras.models import model_from_json
from os.path import join
import matplotlib.pyplot as plt

#%% Define variables
input_size = (128, 128)
label_size = (256, 256)
saved_model_dir_red = '../BSDS500Resized/model/model_red.json'
saved_weights_dir_red = '../BSDS500Resized/model/model_weights_red.h5'
saved_model_dir_blue = '../BSDS500Resized/model/model_blue.json'
saved_weights_dir_blue = '../BSDS500Resized/model/model_weights_blue.h5'
saved_model_dir_green = '../BSDS500Resized/model/model_green.json'
saved_weights_dir_green = '../BSDS500Resized/model/model_weights_green.h5'
output_dir = 'output'
# Define the window size
windowsize_r = 32
windowsize_c = 32

image_dir = '../a10.jpg'


#%% Rescale images 
image = imread(image_dir)
scaled = imresize(image, input_size, 'bicubic')
scaled = imresize(scaled, label_size, 'bicubic')
image =imresize(image, label_size, 'bicubic')


#%% Load model and weights separately due to error in keras
model_red = model_from_json(open(saved_model_dir_red).read())
model_red.load_weights(saved_weights_dir_red)
model_blue = model_from_json(open(saved_model_dir_blue).read())
model_blue.load_weights(saved_weights_dir_blue)
model_green = model_from_json(open(saved_model_dir_green).read())
model_green.load_weights(saved_weights_dir_green)

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


#%%

for r in range(0,input_img.shape[0] - windowsize_r + 1, windowsize_r):
	for c in range(0,input_img.shape[1] - windowsize_c + 1, windowsize_c):
		input_imgs.append(input_img[r:r+windowsize_r,c:c+windowsize_c])
input_imgs = np.asarray(input_imgs)

#%%
output_red = model_red.predict(np.expand_dims(input_imgs[:,:,:,0], axis=3))
output_green = model_green.predict(np.expand_dims(input_imgs[:,:,:,1], axis=3))
output_blue = model_blue.predict(np.expand_dims(input_imgs[:,:,:,2], axis=3))



### Red Channel
i = 0
output_img_red = np.zeros(input_img.shape)
for r in range(0, output_img_red.shape[0] - windowsize_r + 1, windowsize_r):
	for c in range(0,output_img_red.shape[1] - windowsize_c + 1, windowsize_c):
		output_img_red[r:r+windowsize_r,c:c+windowsize_c] = output_red[i,:,:,:]
		i += 1

### Blue Channel
i = 0
output_img_blue = np.zeros(input_img.shape)
for r in range(0, output_img_blue.shape[0] - windowsize_r + 1, windowsize_r):
	for c in range(0,output_img_blue.shape[1] - windowsize_c + 1, windowsize_c):
		output_img_blue[r:r+windowsize_r,c:c+windowsize_c] = output_blue[i,:,:,:]
		i += 1

### Green Channel
i = 0
output_img_green = np.zeros(input_img.shape)
for r in range(0, output_img_green.shape[0] - windowsize_r + 1, windowsize_r):
	for c in range(0,output_img_green.shape[1] - windowsize_c + 1, windowsize_c):
		output_img_green[r:r+windowsize_r,c:c+windowsize_c] = output_green[i,:,:,:]
		i += 1

output_img = np.dstack((output_img_red[:,:,0],output_img_green[:,:,0],output_img_blue[:,:,0]))
print output_img.shape
#%% Format output to 2-D tensor for saving image
#output = np.squeeze(output)

#%% Save images
imsave(join(output_dir, "scaled_image.jpg"), scaled)
imsave(join(output_dir, "output_image.jpg"), output_img)
imsave(join(output_dir, "reference_image.jpg"), image)
