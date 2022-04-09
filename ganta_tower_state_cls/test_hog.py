import matplotlib.pyplot as plt
from skimage import data, exposure
from skimage.feature import hog

image = data.astronaut()

import cv2
filename = '/media/ubuntu/SSD/ganta_patch_classification/train/normal/024_000123_1.000.jpg'
image = cv2.imread(filename)

fd, hog_image = hog(image,
                    orientations=8,
                    pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1),
                    visualize=True,
                    multichannel=True)

import pdb
pdb.set_trace()

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image,
                                                in_range=(0, 10))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4),
                               sharex=True, sharey=True)

ax1.axis('off')
ax2.axis('off')

ax1.imshow(image, cmap=plt.cm.gray)
ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)

ax1.set_title('Input image')
ax2.set_title('HOG')
# plt.show()

plt.savefig('HOG_normal.png')