import numpy as np

def sum_error(array):
    result = np.sum(array ** 2)
    result = np.sqrt(result)
    return result