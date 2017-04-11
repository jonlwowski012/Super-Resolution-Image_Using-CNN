from os import listdir
from os.path import join
from scipy import misc

input_size = (128, 128)
label_size = (256, 256)

input_dir = 'BSDS500/data/images/train'
output_dir = 'BSDS500Resized'

for f in listdir(input_dir):
    input_name = f
    f = join(input_dir, f)

    image = misc.imread(f, flatten=False)

    scaled = misc.imresize(image, input_size, 'bicubic')
    scaled = misc.imresize(scaled, label_size, 'bicubic')

    image = misc.imresize(image, label_size, 'bicubic')

    misc.imsave(join(output_dir, "input",  input_name), scaled)
    misc.imsave(join(output_dir, "label", input_name), image)
