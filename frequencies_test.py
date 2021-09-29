from PIL import Image
import numpy as np
from numpy.fft import rfft2, irfft2
from matplotlib import pyplot as plt


def save_fourier_transforms(ft, low, high, path='./results/frequency', name='test'):
    ft2 = np.zeros_like(ft)
    ft2[low:high, low:high] = ft[low:high, low:high]
    rft = irfft2(ft2)
    result = Image.fromarray(rft).convert('L')
    result.save(f'{path}/{name}_{low}_to_{high - 1}.png')


def main():
    gray_bottle = np.array(Image.open("./figs/bottle.jpg").convert('L'))
    plt.axis('off')
    plt.imshow(gray_bottle, cmap='gray')
    plt.savefig('./results/frequency/gray_bottle.png')
    plt.show()
    ft = rfft2(gray_bottle)
    for i in range(1, 11):
        save_fourier_transforms(ft, 0, i, name='gray_bottle')

    save_fourier_transforms(ft, 0, 20, name='gray_bottle')
    save_fourier_transforms(ft, 0, 50, name='gray_bottle')
    save_fourier_transforms(ft, 0, 100, name='gray_bottle')
    save_fourier_transforms(ft, 0, 1000, name='gray_bottle')


if __name__ == '__main__':
    main()
