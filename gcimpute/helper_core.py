import numpy as np

def observed_part(x): 
	return x[~np.isnan(x)]

def unique_observed(x):
	return np.unique(observed_part(x))

def num_unique(x):
	return len(unique_observed(x))

def dict_values_len(d): 
	return sum([len(v) for v in d.values()])

def merge_dict_arrays(d):
	return np.concatenate(list(d.values()), axis=0)

def project_to_correlation(covariance):
    """
    Projects a covariance to a correlation matrix, normalizing it's diagonal entries. Only checks for diagonal entries to be positive.

    Parameters
    ----------
        covariance: array-like of shape (nfeatures, nfeatures)

    Returns
    -------
        correlation: array-like of shape (nfeatures, nfeatures)
    """
    D = np.diagonal(covariance)
    if any(np.isclose(D, 0)): 
        raise ZeroDivisionError("unexpected zero covariance for the latent Z") 
    D_neg_half = 1.0/np.sqrt(D)
    covariance *= D_neg_half
    correlation = covariance.T * D_neg_half
    return correlation


