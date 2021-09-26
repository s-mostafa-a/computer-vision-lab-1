import numpy as np


def get_kernelized_matrix(matrix: np.array, kernel_shape: tuple, mode: str):
    assert len(matrix.shape) == 2, "The image must be a 2d image!"
    assert len(kernel_shape) == 2, "The kernel must be 2d!"
    assert mode in ("valid", "same"), "The mode parameter can only be `valid` or `same`"
    assert kernel_shape[0] % 2 == 1 and kernel_shape[
        1] % 2 == 1, "kernel shape must be a tuple of odd numbers!"
    padding_i = kernel_shape[0] // 2
    padding_j = kernel_shape[1] // 2
    if mode == "same":
        matrix = np.pad(matrix, [(padding_i,), (padding_j,)])
    shape = tuple([matrix.shape[0] - 2 * padding_i, matrix.shape[1] - 2 * padding_j] +
                  list(kernel_shape))
    kernelized_matrix = np.empty(shape=shape, dtype=matrix.dtype)
    for i in range(padding_i, matrix.shape[0] - padding_i):
        for j in range(padding_j, matrix.shape[1] - padding_j):
            kernelized_matrix[i - padding_i, j - padding_j] = matrix[
                                                              i - padding_i:i + padding_i + 1,
                                                              j - padding_j:j + padding_j + 1]
    return kernelized_matrix
