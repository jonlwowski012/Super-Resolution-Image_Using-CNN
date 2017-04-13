import numpy as np
from scipy.misc import imread, imsave, imresize
from keras.models import model_from_json
from os.path import join

#%% Define variables
input_size = (128, 128)
label_size = (256, 256)
saved_model_dir = 'BSDS500Resized/model'
model_name = ('Rmodel.json', 'Gmodel.json', 'Bmodel.json')
saved_weights_name = ('Rmodel_weights.h5', 'Gmodel_weights.h5', 'Bmodel_weights.h5')
output_dir = 'output'

# Define the window size
windowsize_r = 8
windowsize_c = 8

image_dir = 'a10.jpg'


#%% Rescale images 
image = imread(image_dir)
scaled = imresize(image, input_size, 'bicubic')
scaled = imresize(scaled, label_size, 'bicubic')
image = imresize(image, label_size, 'bicubic')

input_img = scaled.copy()
output_img = np.zeros(input_img.shape)


#%% Loop over all channels
for i in range(3):
    
    # Load model and weights separately due to error in keras for custom loss functions
    model = model_from_json(open(join(saved_model_dir, model_name[i])).read())
    model.load_weights(join(saved_model_dir, saved_weights_name[i]))

    # Separate by channel
    input_chan = input_img[:,:,i]
    input_chan = np.expand_dims(input_chan, axis=3)
    model_input = []
    model_output = np.zeros(input_chan.shape)
    
    # Splice image channel
    for r in range(0,input_chan.shape[0] - windowsize_r + 1, windowsize_r):
    	for c in range(0,input_chan.shape[1] - windowsize_c + 1, windowsize_c):
    		model_input.append(input_chan[r:r+windowsize_r,c:c+windowsize_c])
    model_input = np.asarray(model_input)

    # Predict output
    output = model.predict(model_input)
    
    
    # Reconstruct channel from slices
    j = 0
    for r in range(0, model_output.shape[0] - windowsize_r + 1, windowsize_r):
    	for c in range(0,model_output.shape[1] - windowsize_c + 1, windowsize_c):
    		model_output[r:r+windowsize_r,c:c+windowsize_c] = output[j,:,:]
    		j += 1

    # Add reconstructed output to channel index
    model_output = np.squeeze(model_output)
    output_img[:,:,i] = model_output


#%% Save images
imsave(join(output_dir, "scaled_image.jpg"), scaled)
imsave(join(output_dir, "output_image.jpg"), output_img)
imsave(join(output_dir, "reference_image.jpg"), image)