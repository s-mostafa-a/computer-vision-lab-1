from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import expand, contract, block_matrix

# im = np.array(Image.open('./figs/bird.bmp'))
im = np.array(
    [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20],
     [21, 22, 23, 24, 25]])
half_neigh = 1
kol_neigh = half_neigh * 2 + 1
print(im.shape)
plt.axis('off')
plt.imshow(im)
plt.show()
first = im[:, :]
print(first.shape)
plt.axis('off')
plt.imshow(first)
plt.show()
ex = expand(first, half_neigh, 'same')
print(ex.shape)
plt.axis('off')
plt.imshow(ex)
plt.show()
final_shape = tuple(list(first.shape) + [kol_neigh, kol_neigh])
sm = ex.reshape(final_shape)
# sm = block_matrix(ex, (kol_neigh, kol_neigh))
after_zarb = sm * np.ones((1, 1, kol_neigh, kol_neigh))
print(after_zarb.shape)
# sm = contract(ex, half_neigh)
