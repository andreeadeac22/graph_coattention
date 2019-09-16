"""
	File used to train the networks.
"""
import csv
import os
import pprint
import time
import random
import logging
import argparse
from prettytable import PrettyTable

import numpy as np
from sklearn import metrics
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data


def main():
	parser = argparse.ArgumentParser()

	# Directory to resume training from
	parser.add_argument('--trained_setting_pkl', default=None,
						help='Load trained model from setting pkl')

	# Directory containing precomputed training data split.
	parser.add_argument('--data_pkl', default=None,
						help="Fold data path, e.g. ./data/decagon/folds/")

	parser.add_argument('-f', '--fold', default='1/10', type=str,
	                    help="Which fold to run, format x/total")

	# Dirs
	parser.add_argument('--model_dir', default='./exp_trained')
	parser.add_argument('--result_dir', default='./exp_results')
	parser.add_argument('--setting_dir', default='./exp_settings')
	parser.add_argument('-mm', '--memo', help='Memo for experiment', default='default')

	parser.add_argument('-b', '--batch_size', type=int, default=128)

	parser.add_argument('-d_h', '--d_hid', type=int, default=32)
	parser.add_argument('-d_a', '--d_atom_feat', type=int, default=3)
	parser.add_argument('-n_p', '--n_prop_step', type=int, default=3)
	parser.add_argument('-n_h', '--n_attention_head', type=int, default=1)
	parser.add_argument('-score', '--score_fn', default='trans', const='trans',
						nargs='?', choices=['trans', 'factor'])

	parser.add_argument('-dbg', '--debug', action='store_true')
	parser.add_argument('-e', '--n_epochs', type=int, default=10000)
	parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
	parser.add_argument('-l2', '--l2_lambda', type=float, default=0)
	parser.add_argument('-drop', '--dropout', type=float, default=0.1)
	parser.add_argument('--patience', type=int, default=32)
	parser.add_argument('--seed', type=int, default=777)
	parser.add_argument('-nr', '--train_neg_pos_ratio', type=int, default=1)
	parser.add_argument('-tr', '--transR', action='store_true')
	parser.add_argument('-th', '--transH', action='store_true')

	opt = parser.parse_args()

	opt.fold_i, opt.n_fold = map(int, opt.fold.split('/'))
	assert 0 < opt.fold_i <= opt.n_fold

	is_resume_training = opt.trained_setting_pkl is not None
	if is_resume_training:
		logging.info('Resume training from', opt.trained_setting_pkl)
		opt = np.load(open(opt.trained_setting_pkl, 'rb')).item()
	else:
		opt = post_parse_args(opt)
		setup_directories(opt)
		# save the dataset split in setting dictionary
		pprint.pprint(vars(opt))
		# save the setting
		logging.info('Related data will be saved with prefix: %s', opt.exp_prefix)



if __name__ == "__main__":
	main()
