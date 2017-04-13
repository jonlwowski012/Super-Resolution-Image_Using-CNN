from os import listdir
from os.path import join
from scipy import misc
import os

input_size = (128, 128)
label_size = (256, 256)

input_dir = 'Caltech256'
output_dir = 'Caltech256_Resized'
for root, dirs, files in os.walk(input_dir):
	for name in files:
		if name.endswith((".png", ".jpg")):
			input_name = name
			f = root+"/"+name

			image = misc.imread(f, flatten=False)

			scaled = misc.imresize(image, input_size, 'bicubic')
			scaled = misc.imresize(scaled, label_size, 'bicubic')

			image = misc.imresize(image, label_size, 'bicubic')

			misc.imsave(join(output_dir, "input",  input_name), scaled)
			misc.imsave(join(output_dir, "label", input_name), image)
