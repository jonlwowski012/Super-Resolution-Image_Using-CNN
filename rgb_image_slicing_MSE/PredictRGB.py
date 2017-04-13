import numpy as np
from scipy.misc import imread, imsave, imresize
from keras.models import model_from_json
from os.path import join


#%% Define performance metrics
def mse(y_true, y_pred):
    MSE = np.mean(np.square(y_pred - y_true))   
    return MSE

def psnr(y_true, y_pred):
    max_pixel_val = 255
    MSE = np.mean(np.square(y_pred - y_true))
    PSNR = (10. * np.log10(np.square(max_pixel_val) / MSE))
    return PSNR 


#%% Define variables
input_size = (128, 128)
label_size = (256, 256)
saved_model_dir = 'msemodel/64x64'
model_name = ('Rmodel.json', 'Gmodel.json', 'Bmodel.json')
saved_weights_name = ('Rmodel_weights.h5', 'Gmodel_weights.h5', 'Bmodel_weights.h5')
output_dir = 'output'

# Define the window size
windowsize_r = 64
windowsize_c = 64

image_dir = 'lena.bmp'


#%% Rescale images
image = imread(image_dir)
scaled = imresize(image, input_size, 'bicubic')
scaled = imresize(scaled, label_size, 'bicubic')
ref_img = (imresize(image, label_size, 'bicubic'))

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
output_img = output_img.astype('uint8')
imsave(join(output_dir, "scaled_image.jpg"), scaled)
imsave(join(output_dir, "output_image.jpg"), output_img)
imsave(join(output_dir, "reference_image.jpg"), ref_img)

#%% 
print('Scaled MSE:', mse(ref_img, scaled), 'PSNR:', psnr(ref_img, scaled))
print('Output MSE:', mse(ref_img, output_img), 'PSNR:', psnr(ref_img, output_img))
