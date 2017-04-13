import numpy as np
import keras.backend as K
from os import listdir
from os.path import join
from scipy import misc
from keras.models import Model
from keras.layers import Input, Conv2D
from keras.layers import merge
from keras.layers.merge import add
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, EarlyStopping
from keras.initializers import RandomNormal
from keras_contrib.losses import DSSIMObjective


#%%Define paths and variables
input_dir = 'BSDS500Resized/input'
label_dir = 'BSDS500Resized/label'
model_dir = 'BSDS500Resized/model'
model_name = ('Rmodel.json', 'Gmodel.json', 'Bmodel.json')
model_weights_name = ('Rmodel_weights.h5', 'Gmodel_weights.h5', 'Bmodel_weights.h5')

epochs=20
batch_size=100
max_pixel_val = float(255)


# Define the window size
windowsize_r = 64
windowsize_c = 64

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

X = np.asarray
y = np.asarray

#%% Shave off some samples
#X = X[0:192000]
#y = y[0:192000]

#%% Define model

# Define weights and bias intializers
kernel_init = RandomNormal(mean=0.0, stddev=0.001)
bias_init = RandomNormal(mean=0.0, stddev=0.0)

# Model
inputs = Input(shape=(windowsize_r, windowsize_c, 1))
x = Conv2D(64, (9, 9), input_shape=(windowsize_r, windowsize_c, 1), activation='relu',
           kernel_initializer=kernel_init, bias_initializer=bias_init, padding='same')(inputs)
x = Conv2D(64, (5, 5), activation='relu', kernel_initializer=kernel_init,
           bias_initializer=bias_init,padding='same')(x)
x = Conv2D(1, (5, 5), kernel_initializer=kernel_init, bias_initializer=bias_init, padding='same')(x)
x = merge(([inputs, x]), mode='sum')
model = Model(inputs=inputs, outputs=x)

# Define SSIM Loss
SSIM = DSSIMObjective(k1=0.01, k2=0.03, kernel_size=2, max_value=1.0)

# Log epoch results to CSV file
csv_logger = CSVLogger('training.log')

# Stop early if loss converges
early_stop = EarlyStopping(monitor='loss', patience=2)

# Define PSNR performance metric Note: max_pixel_value = 255 for uint8
# PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
def psnr(y_true, y_pred):
    return (20. * K.log(max_pixel_val)) - (10. * K.log(K.mean(K.square(y_pred - y_true))))

# Compile model
model.compile(optimizer=Adam(lr=0.001), loss=SSIM, metrics=['mse', psnr])



#%% Loop for R G and B channels
for i in range(3):
    Xtrain = X[:,:,:,i]
    ytrain = y[:,:,:,i]

    Xtrain = np.expand_dims(Xtrain, axis=3)
    ytrain = np.expand_dims(ytrain, axis=3)

    # Train model
    model.fit(Xtrain, ytrain, epochs=epochs, batch_size=batch_size, callbacks=[csv_logger, early_stop])

    # Save parameters
    config = model.to_json()
    open(join(model_dir, model_name[i]), "w").write(config)
    model.save_weights(join(model_dir, model_weights_name[i]))
