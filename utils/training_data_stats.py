import pickle
import numpy as np
import torch


def get_qm9_stats():
	"""
	min
	[ 0.0000000e+00  6.3099999e+00 -4.2860001e-01 -1.7500000e-01
	  3.7599999e-02  1.9000200e+01  1.5951000e-02 -7.1456805e+02
	 -7.1456018e+02 -7.1455920e+02 -7.1460211e+02  6.0019999e+00]

	max
	[ 2.5202200e+01  1.4353000e+02 -1.0170000e-01  1.9350000e-01
	  6.2210000e-01  3.3747532e+03  2.7394399e-01 -5.6525887e+01
	 -5.6523026e+01 -5.6522083e+01 -5.6544960e+01  4.6969002e+01]

	mean
	[ 2.7038786e+00  7.5196838e+01 -2.3998778e-01  1.1123519e-02
	  2.5111273e-01  1.1894105e+03  1.4853235e-01 -4.1154922e+02
	 -4.1153308e+02 -4.1153122e+02 -4.1160294e+02  3.1599964e+01]

	std
	[1.5315679e+00 8.1900263e+00 2.2168942e-02 4.6939783e-02 4.7541428e-02
	 2.7926233e+02 3.3273432e-02 4.0096535e+01 4.0096313e+01 4.0096313e+01
	 4.0097050e+01 4.0638595e+00]
	"""
	train_labels_dict = pickle.load(open("data/qm9/dsgdb9nsd/folds/train_labels.npy", "rb"))
	labels = list(train_labels_dict.values())

	labels = torch.Tensor(np.stack(labels))
	print(labels.shape)

	minima = torch.min(labels, 0)[0]
	maxima = torch.max(labels, 0)[0] # first is values, second indices

	mean = torch.mean(labels, 0)
	std = torch.std(labels, 0)

	return minima, maxima, mean, std


