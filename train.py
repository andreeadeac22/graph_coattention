"""
	File used to train the networks.
"""
import os
import pprint
import random
import logging
import argparse
import pickle

import numpy as np
import torch
import torch.utils.data

from utils.file_utils import setup_running_directories, save_experiment_settings

def post_parse_args(opt):
	# Set the random seed manually for reproducibility.
	random.seed(opt.seed)
	np.random.seed(opt.seed)
	torch.manual_seed(opt.seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(opt.seed)
	if not hasattr(opt, 'exp_prefix'):
		opt.exp_prefix = opt.memo + '-cv_{}_{}'.format(opt.fold_i, opt.n_fold)
	if opt.debug:
		opt.exp_prefix = 'dbg-{}'.format(opt.exp_prefix)
	if not hasattr(opt, 'global_step'):
		opt.global_step = 0

	opt.best_model_pkl = os.path.join(opt.model_dir, opt.exp_prefix + '.pth')
	opt.result_csv_file = os.path.join(opt.result_dir, opt.exp_prefix + '.csv')
	return opt


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset', metavar='D', type=str.lower,
	                    choices=['qm9', 'decagon'],
	                    help='Name of dataset to used for training [QM9,DECAGON]')

	# Directory to resume training from
	parser.add_argument('--trained_setting_pkl', default=None,
						help='Load trained model from setting pkl')

	#parser.add_argument('input_data', default=None,
	#                    help='Loading additional input data (n_atoms,...)'
	#                        'e.g. ./data/decagon/folds/input_data.npy')

	# Directory containing precomputed training data split.
	parser.add_argument('input_data_path', default=None,
						help="Input data path, e.g. ./data/decagon/ "
						     "or ./data/qm9/dsgdb9nsd/")

	#parser.add_argument('--qm9_labels', default=
	#	'./data/qm9/dsgdb9nsd/drug.labels.jsonl')
	#parser.add_argument('--ddi_data', default=
	#	'./data/decagon/bio-decagon-combo.csv')

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
	parser.add_argument('--seed', type=int, default=1337)
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
		setup_running_directories(opt)
		# save the dataset split in setting dictionary
		pprint.pprint(vars(opt))
		# save the setting
		logging.info('Related data will be saved with prefix: %s', opt.exp_prefix)

	assert os.path.exists(opt.input_data_path + "folds/")

	if "qm9" in opt.dataset:
		data_opt = np.load(open(opt.input_data_path + "input_data.npy",'rb')).item()

		opt.n_atom_type = data_opt.n_atom_type
		opt.n_bond_type = data_opt.n_bond_type


		opt.train_graph_dict = pickle.load(open(opt.input_data_path + "folds/" + "train_graphs.npy", "rb"))
		opt.train_labels_dict = pickle.load(open(opt.input_data_path + "folds/" + "train_labels.npy", "rb"))

		opt.valid_graph_dict = pickle.load(open(opt.input_data_path + "folds/" + "valid_graphs.npy", "rb"))
		opt.valid_labels_dict = pickle.load(open(opt.input_data_path + "folds/" + "valid_labels.npy", "rb"))

		opt.test_graph_dict = pickle.load(open(opt.input_data_path + "folds/" + "test_graphs.npy", "rb"))
		opt.test_labels_dict = pickle.load(open(opt.input_data_path + "folds/" + "test_labels.npy", "rb"))



	if "decagon" in opt.dataset:
		data_opt = np.load(open(opt.input_data_path + "input_data.npy", 'rb')).item()
		opt.n_atom_type = data_opt.n_atom_type
		opt.n_bond_type = data_opt.n_bond_type
		opt.n_side_effect = data_opt.n_side_effect

		graph_dict = pickle.load(open(opt.input_data_path + "folds/" + "1fold.npy", "rb"))
		for se in graph_dict['pos']:
			print(se)
			print(graph_dict['pos'][se])
			break

		"""
		assert data_opt.n_side_effect == len(opt.side_effects)

		data_opt = np.load(open(opt.data_pkl, 'rb')).item()
		opt.ddi_data = data_opt.ddi_data
		opt.drug_data = data_opt.drug_data
		opt.drug_dict = data_opt.drug_dict
		opt.side_effects = data_opt.side_effects
		opt.side_effect_idx_dict = data_opt.side_effect_idx_dict
		"""

	save_experiment_settings(opt)


if __name__ == "__main__":
	main()
