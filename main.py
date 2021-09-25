from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

im = np.array(Image.open('./figs/bird.bmp'))
im = np.array(
    [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
print(im.shape)
plt.axis('off')
plt.imshow(im)
plt.show()
