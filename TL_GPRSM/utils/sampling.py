import numpy as np

def latin_hypercube_sampling(sampling_num:int, dim:int, center:bool=False, seed:int=None)->np.array:
    """Latin hypercube sampling

    Args:
        sampling_num (int): number of sampling
        dim (int): dimension of sampling
        center (bool, optional): is center. Defaults to False.
        seed (int, optional): random seed. Defaults to None.
    Returns:
        numpy.ndarray: sampling points (number of samples x number of features)
    """
    if seed is not None:
        np.random.seed(seed)
    sampling_points = np.zeros(shape=(sampling_num, dim))
    for i in range(dim):
        sampling_points[:,i] = np.random.permutation(sampling_num)
    if center:
        sampling_points = (sampling_points + 0.5) / sampling_num
    else:
        sampling_points = (sampling_points + np.random.rand(sampling_num, dim)) / sampling_num
    return sampling_points

def random_sampling(sampling_num:int, dim:int, seed:int=None)->np.array:
    """Random sampling

    Args:
        sampling_num (int): number of sampling
        dim (int): dimension of sampling
        seed (int, optional): random seed. Defaults to None.
    Returns:
        numpy.ndarray: sampling points (number of samples x number of features)
    """    
    if seed is not None:
        np.random.seed(seed)
    sampling_points = np.random.rand(sampling_num, dim)
    return sampling_points

def uniform_scaling(sampling_points, scale_mins, scale_maxs):
    """Uniform scaling

    Args:
        sampling_points (np.array): sampling points (number of samples x number of features)
        scale_mins (list[float,...]): min vals of each feature
        scale_maxs (list[float,...]): max vals of each feature

    Returns:
        np.array: scaled sampling points (number of samples x number of features)
    """
    sampling_points = sampling_points * (scale_maxs - scale_mins) + scale_mins
    return sampling_points

if __name__=="__main__":
    import matplotlib.pyplot as plt
    sample = latin_hypercube_sampling(1000, 1, False)
    sample = (sample-np.mean(sample))/np.std(sample)
    plt.hist(sample, bins=20)
    plt.savefig("test.png")