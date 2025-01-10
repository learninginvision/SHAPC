import numpy as np
import torch
from PIL import Image
from scipy import ndimage

def kl_divergence(target, input, epsilon=1e-10,axis=0):
    p_smoothed = (target + epsilon) / np.sum(target + epsilon,axis=axis,keepdims=True)
    q_smoothed = (input + epsilon) / np.sum(input + epsilon,axis=axis,keepdims=True)
    return np.sum(p_smoothed * np.log(p_smoothed / q_smoothed))

def normalization(array, axis=0, rank='max'):
    if rank == 'max':
        min_array = np.min(array, axis=axis, keepdims=True)
        relativive_array = array - min_array
        result = relativive_array / np.sum(relativive_array, axis=axis, keepdims=True)
    elif rank == 'min':
        max_array = np.max(array, axis=axis, keepdims=True)
        relativive_array = array - max_array
        result = relativive_array / np.sum(relativive_array, axis=axis, keepdims=True)
    elif rank == 'max-min':
        max_array = np.max(array, axis=axis, keepdims=True)
        min_array = np.min(array, axis=axis, keepdims=True)
        result = array/(max_array-min_array)
    else:
        assert False, 'Argument rank should be "max" or "min"'
    return result

def mask_array(array, percent=0.5, rank='max'):
    '''
    Array: width x height 
    '''
    total_elements = array.shape[0] * array.shape[1]
    flattened_array = array.flatten()
    if rank=='max':
        sort_index = np.argsort(flattened_array)[::-1]
    elif rank=='min':
        sort_index = np.argsort(flattened_array)
    else:
        assert False, 'Argument rank should be "max" or "min"'
    threshold_index = int(total_elements * percent)    
    result = np.zeros(total_elements)
    _sort_index = sort_index[:threshold_index]
    # rows = np.arange(flattened_array.shape[0])
    # result[rows[:, np.newaxis], _sort_index] = 1
    
    for i in _sort_index:
        result[i] = 1
    arr = np.reshape(result, array.shape)
    return arr


def IoU(mask1, mask2):
        
    int_count = np.bitwise_and(mask1, mask2)
    uni_count = np.bitwise_or(mask1, mask2)
    result = np.sum(int_count) / np.sum(uni_count)

    return result, int_count, uni_count

def mask_classes(outputs: torch.Tensor, N_CLASSES_PER_TASK, N_TASKS, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * N_CLASSES_PER_TASK:
               N_TASKS * N_CLASSES_PER_TASK] = -float('inf')

def nonzero_elements(array):
    return array[array != 0]

def FC(input, target, mask_int, mask_uni):
    _int_region_i = nonzero_elements(input*mask_int)
    _int_region_t = nonzero_elements(target*mask_int)
    numerator = np.sum(np.exp(-np.abs(_int_region_t-_int_region_i)))
    _uni_region_i = nonzero_elements(input*mask_uni)
    _uni_region_t = nonzero_elements(target*mask_uni)
    denominator = np.sum(np.exp(-np.abs(_uni_region_t-_uni_region_i)))    
    return numerator/denominator

if '__main__' == __name__:    
    arr = np.array([[1,2,3,4,5], [2,3,4,5,6]])
