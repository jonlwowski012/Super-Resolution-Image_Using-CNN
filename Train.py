import numpy as np
import keras.backend as K
from os import listdir
from os.path import join
from scipy import misc
from keras.models import Model
from keras.layers import Input, Conv2D
from keras.optimizers import Adam
from keras.callbacks import CSVLogger
from keras_contrib.losses import DSSIMObjective


#%% Define paths and variables
input_dir = 'BSDS500Resized/input'
label_dir = 'BSDS500Resized/label'
model_dir = 'BSDS500Resized/model'

epochs=10
batch_size=12000
max_pixel_val = float(255)


# Define the window size
windowsize_r = 8
windowsize_c = 8


#%% Import images
X = []
for f in listdir(input_dir):
	test_image = np.asarray(misc.imread(join(input_dir, f)))
	# Crop out the window and calculate the histogram
	for r in range(0,test_image.shape[0] - windowsize_r, windowsize_r):
    		for c in range(0,test_image.shape[1] - windowsize_c, windowsize_c):
        		X.append(test_image[r:r+windowsize_r,c:c+windowsize_c])
y = []
for f in listdir(label_dir):
	test_image = np.asarray(misc.imread(join(label_dir, f)))
	# Crop out the window and calculate the histogram
	for r in range(0,test_image.shape[0] - windowsize_r, windowsize_r):
    		for c in range(0,test_image.shape[1] - windowsize_c, windowsize_c):
        		y.append(test_image[r:r+windowsize_r,c:c+windowsize_c])

X = np.asarray(X)
y = np.asarray(y)

#%%
X = X[0:480000]
y = y[0:480000]
#X = np.array([misc.imread(join(input_dir, f)) for f in listdir(input_dir)])
#y = np.array([misc.imread(join(label_dir, f)) for f in listdir(label_dir)])

#print(X.shape)

#X = np.expand_dims(X, axis=0)
#y = np.expand_dims(y, axis=0)
#X = np.transpose(X, (1, 2, 3, 0))
#y = np.transpose(y, (1, 2, 3, 0))


#%% Define model
inputs = Input(shape=(windowsize_r, windowsize_c, 3))
x = Conv2D(64, (9, 9), input_shape=(windowsize_r, windowsize_c, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
x = Conv2D(32, (1, 1), activation='relu', kernel_initializer='he_normal', padding='same')(x)
x = Conv2D(3, (5, 5), kernel_initializer='he_normal', padding='same')(x)
model = Model(inputs=inputs, outputs=x)


#%% Define SSIM Loss
SSIM = DSSIMObjective(k1=0.01, k2=0.03, kernel_size=2, max_value=1.0)


#%% Define PSNR performance metric Note: max_pixel_value = 255 for uint8
# PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
def psnr(y_true, y_pred):
    return (20. * K.log(max_pixel_val)) - (10. * K.log(K.mean(K.square(y_pred - y_true))))

#%% Log epoch results to CSV file
csv_logger = CSVLogger('training.log')

#%% Compile and train model
model.compile(optimizer=Adam(lr=0.001), loss=SSIM, metrics=['mse', psnr])
model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=[csv_logger])


#%% Save parameters
config = model.to_json()
open(join(model_dir, "model.json"), "w").write(config)
model.save_weights(join(model_dir, 'model_weights.h5'))