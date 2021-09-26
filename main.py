from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import get_kernelized_matrix

image = np.array(Image.open('./figs/bird.bmp'))[:, :, 0]
# image = np.array(
#     [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20],
#      [21, 22, 23, 24, 25]])
plt.axis('off')
plt.imshow(image)
plt.show()

kernel = np.ones((3, 3))
kernel = kernel / np.sum(kernel)

kernelized_matrix = get_kernelized_matrix(matrix=image, kernel_shape=kernel.shape, mode='same')
expanded_kernel = np.expand_dims(np.expand_dims(kernel, axis=0), axis=0)

low_path_filtered = np.sum((kernelized_matrix * expanded_kernel), axis=(2, 3))
plt.axis('off')
plt.imshow(low_path_filtered)
plt.show()
