import numpy as np 
import pandas as pd
from numpy.linalg import eig

def create_cat_index_map(cat_index_level, detailed_return = False):
	'''
	Build a dict indicating categorical dimensions from a vector of the number of categories for all variables and 
	the input has None at non-categorical variables
	'''
	last = -1
	starts = []
	noncat_index_map = {}
	cat_index_map = {}
	for i,x in enumerate(cat_index_level):
		if x is None:
			noncat_index_map[i] = last + 1
			last += 1
		else:
			cat_index_map[i] = last + 1 + np.arange(x)
			starts.append(last+1)
			last += x
	if detailed_return:
		return {'cat_index_map':cat_index_map, 'noncat_index_map':noncat_index_map, 'starts':starts}
	else:
		return cat_index_map

def project_corr_with_cat(corr, cat_index_map : dict, eps = 1e-5):
	'''
	Project a copula correlation matrix to be the identity at each categorical block
	'''
	p = len(corr)
	A = np.identity(p)
	for i,index in cat_index_map.items():
		m_index = np.ix_(index, index)
		eigen_values, eigen_vec = eig(corr[m_index])
		if eigen_values.min() < eps:
			print('Projection skipped: small eigenvalue in a categorical block')
			A[m_index] = np.identity(len(index))
		else:
			m = (eigen_vec * np.sqrt(1/eigen_values)) @ eigen_vec.T
			A[m_index] = m

	corr = A @ corr @ A.T

	for i,index in cat_index_map.items():
		corr[np.ix_(index, index)] = np.identity(len(index))

	return corr

