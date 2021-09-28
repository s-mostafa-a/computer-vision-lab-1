import numpy as np


def _assertion_checks(image, kernel, mode):
    assert len(image.shape) == 2, "The image must be a 2d image!"
    assert len(kernel.shape) == 2, "The kernel must be 2d!"
    assert mode in ("valid", "same"), "The mode parameter can only be `valid` or `same`"
    assert kernel.shape[0] % 2 == 1 and kernel.shape[
        1] % 2 == 1, "kernel shape must be a tuple of odd numbers!"


def _get_paddings_and_new_image(image, kernel_shape, mode):
    padding_i = kernel_shape[0] // 2
    padding_j = kernel_shape[1] // 2
    if mode == "same":
        image = np.pad(image, [(padding_i,), (padding_j,)])
    return padding_i, padding_j, image


def low_pass_filter_using_broadcast(image: np.array, kernel: np.array, mode: str = 'same'):
    _assertion_checks(image=image, kernel=kernel, mode=mode)
    padding_i, padding_j, image = _get_paddings_and_new_image(image=image,
                                                              kernel_shape=kernel.shape, mode=mode)
    shape = tuple([image.shape[0] - 2 * padding_i, image.shape[1] - 2 * padding_j] +
                  list(kernel.shape))
    multi_dim_image = np.empty(shape=shape, dtype=image.dtype)
    for i in range(padding_i, image.shape[0] - padding_i):
        for j in range(padding_j, image.shape[1] - padding_j):
            multi_dim_image[i - padding_i, j - padding_j] = image[
                                                            i - padding_i:i + padding_i + 1,
                                                            j - padding_j:j + padding_j + 1]

    expanded_kernel = np.expand_dims(np.expand_dims(kernel, axis=0), axis=0)

    final_image = np.sum((multi_dim_image * expanded_kernel), axis=(2, 3))
    return final_image


def low_pass_filter_without_broadcast(image: np.array, kernel: np.array, mode: str = 'same'):
    _assertion_checks(image=image, kernel=kernel, mode=mode)
    padding_i, padding_j, image = _get_paddings_and_new_image(image=image,
                                                              kernel_shape=kernel.shape, mode=mode)
    shape = (image.shape[0] - 2 * padding_i, image.shape[1] - 2 * padding_j)
    final_image = np.empty(shape=shape, dtype=image.dtype)
    for i in range(padding_i, image.shape[0] - padding_i):
        for j in range(padding_j, image.shape[1] - padding_j):
            summation = np.sum(
                image[i - padding_i:i + padding_i + 1, j - padding_j:j + padding_j + 1] * kernel)
            final_image[i - padding_i, j - padding_j] = summation
    return final_image


def gaussian_kernel(cutoff_frequency=1.):
    kernel_size = int(cutoff_frequency * 4 + 1)
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(cutoff_frequency))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def box_kernel(size=3):
    kernel = np.ones((size, size))
    return kernel / np.sum(kernel)
