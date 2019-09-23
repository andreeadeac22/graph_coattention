"""
	File used to train the networks.
"""
import os
import csv
import pprint
import random
import logging
import argparse
import pickle
from tqdm import tqdm
from sklearn import metrics

import numpy as np
import torch
import torch.optim as optim
import torch.utils.data

from qm9_dataset import QM9Dataset, qm9_collate_batch
from ddi_dataset import ddi_collate_paired_batch, \
	PolypharmacyDataset, ddi_collate_batch
from utils.file_utils import setup_running_directories, \
				save_experiment_settings
from utils.functional_utils import combine

from utils.qm9_utils import build_qm9_dataset, \
				qm9_train_epoch, qm9_valid_epoch
from utils.ddi_utils import ddi_train_epoch, ddi_valid_epoch


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


def prepare_qm9_dataloaders(opt):
	train_loader = torch.utils.data.DataLoader(
		QM9Dataset(
			graph_dict = opt.graph_dict,
			pairs_dataset = opt.train_dataset),
		num_workers = 2,
		batch_size = opt.batch_size,
		collate_fn = qm9_collate_batch,
		shuffle = True)
	valid_loader = torch.utils.data.DataLoader(
		QM9Dataset(
			graph_dict=opt.graph_dict,
			pairs_dataset = opt.valid_dataset),
		num_workers = 2,
		batch_size = opt.batch_size,
		collate_fn = qm9_collate_batch)
	return train_loader, valid_loader


def prepare_ddi_dataloaders(opt):
	train_loader = torch.utils.data.DataLoader(
		PolypharmacyDataset(
			drug_structure_dict= opt.graph_dict,
			se_idx_dict = opt.side_effect_idx_dict,
			se_pos_dps = opt.train_dataset['pos'],
			#TODO: inspect why I'm not just fetching opt.train_dataset['neg']
			negative_sampling=True,
			negative_sample_ratio=opt.train_neg_pos_ratio,
			paired_input=True,
			n_max_batch_se=10),
		num_workers=2,
		batch_size=opt.batch_size,
		collate_fn=ddi_collate_paired_batch,
		shuffle=True)

	valid_loader = torch.utils.data.DataLoader(
		PolypharmacyDataset(
			drug_structure_dict = opt.graph_dict,
			se_idx_dict = opt.side_effect_idx_dict,
			se_pos_dps = opt.valid_dataset['pos'],
			se_neg_dps = opt.valid_dataset['neg'],
			n_max_batch_se=1),
		num_workers=2,
		batch_size=opt.batch_size,
		collate_fn=lambda x: ddi_collate_batch(x, return_label=True))
	return train_loader, valid_loader


def train_epoch(model, data_train, optimizer, averaged_model, device, opt):
	if opt.dataset == "qm9":
		return qm9_train_epoch(model, data_train, optimizer, averaged_model, device, opt)
	else:
		return ddi_train_epoch(model, data_train, optimizer, averaged_model, device, opt)


def valid_epoch(model, data_valid, device, opt):
	if opt.dataset == "qm9":
		return qm9_valid_epoch(model, data_valid, device, opt)
	else:
		return ddi_valid_epoch(model, data_valid, device, opt)


def train(model, datasets, device, opt):

	data_train, data_valid = datasets
	optimizer = optim.Adam(
		model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_lambda)

	with open(opt.result_csv_file, 'w') as csv_file:
		csv_writer = csv.writer(csv_file)
		if opt.dataset == "decagon":
			csv_writer.writerow(['train_loss', 'auroc_valid'])
		if opt.dataset == "qm9":
			csv_writer.writerow(['train_loss', 'auroc_valid', 'individual_maes'])


	ddi_best_valid_perf = 0
	qm9_best_valid_perf = 10
	waited_epoch = 0
	averaged_model = model.state_dict()
	for epoch_i in range(opt.n_epochs):
		print()
		print('Epoch ', epoch_i)

		# ============= Training Phase =============
		train_loss, elapse, averaged_model = \
			train_epoch(model, data_train, optimizer, averaged_model, device, opt)
		logging.info('  Loss:       %5f, used time: %f min', train_loss, elapse)

		# ============= Validation Phase =============

		# Load the averaged model weight for validation
		updated_model = model.state_dict() # validation start
		model.load_state_dict(averaged_model)

		valid_perf, elapse = valid_epoch(model, data_valid, device, opt)
		# AUROC is in fact MAE for QM9
		valid_auroc = valid_perf['auroc']
		if opt.dataset == "qm9":
			individual_maes = valid_perf['individual_maes']
		logging.info('  Validation: %5f, used time: %f min', valid_auroc, elapse)
		#print_performance_table({k: v for k, v in valid_perf.items() if k != 'threshold'})

		# Load back the trained weight
		model.load_state_dict(updated_model) # validation end

		# early stoppingf

		if (opt.dataset == "decagon" and valid_auroc >  ddi_best_valid_perf) \
				or (opt.dataset == "qm9" and valid_auroc < qm9_best_valid_perf):
			logging.info('  --> Better validation result!')
			waited_epoch = 0
			torch.save(
				{'global_step': opt.global_step,
				 'model':averaged_model,
				 'threshold': valid_perf['threshold']},
				opt.best_model_pkl)
		else:
			if waited_epoch < opt.patience:
				waited_epoch += 1
				logging.info('  --> Observing ... (%d/%d)', waited_epoch, opt.patience)
			else:
				logging.info('  --> Saturated. Break the training process.')
				break

		# ============= Bookkeeping Phase =============
		# Keep the validation record
		if opt.dataset == "decagon":
			ddi_best_valid_perf = max(valid_auroc, ddi_best_valid_perf)
		if opt.dataset == "qm9":
			qm9_best_valid_perf = min(valid_auroc, qm9_best_valid_perf)

		# Keep all metrics in file
		with open(opt.result_csv_file, 'a') as csv_file:
			csv_writer = csv.writer(csv_file)
			if opt.dataset == "decagon":
				csv_writer.writerow([train_loss, valid_auroc])
			if opt.dataset == "qm9":
				csv_writer.writerow([train_loss, valid_auroc, individual_maes])



def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset', metavar='D', type=str.lower,
	                    choices=['qm9', 'decagon'],
	                    help='Name of dataset to used for training [QM9,DECAGON]')

	# Directory to resume training from
	parser.add_argument('--trained_setting_pkl', default=None,
						help='Load trained model from setting pkl')

	# Directory containing precomputed training data split.
	parser.add_argument('input_data_path', default=None,
						help="Input data path, e.g. ./data/decagon/ "
						     "or ./data/qm9/dsgdb9nsd/")

	parser.add_argument('-f', '--fold', default='1/10', type=str,
	                    help="Which fold to test on, format x/total")

	parser.add_argument('--qm9_pairing_repetitions', default = 1, type=int,
	                    help="How many times to pair each molecule with a random molecule")
	parser.add_argument('--qm9_output_feat', default=12, type=int,
	                    help="How many features need to be predicted for qm9")

	# Dirs
	parser.add_argument('--model_dir', default='./exp_trained')
	parser.add_argument('--result_dir', default='./exp_results')
	parser.add_argument('--setting_dir', default='./exp_settings')
	parser.add_argument('-mm', '--memo', help='Memo for experiment', default='default')

	parser.add_argument('-b', '--batch_size', type=int, default=128)

	parser.add_argument('-d_h', '--d_hid', type=int, default=32)
	parser.add_argument('-d_readout', '--d_readout', type=int, default=32)

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

	parser.add_argument('--qm9_normalise', default="scaled",
	                    help="How to normalise qm9 data e.g. scaled or standardised")

	opt = parser.parse_args()

	assert opt.batch_size % opt.qm9_pairing_repetitions == 0

	opt.fold_i, opt.n_fold = map(int, opt.fold.split('/'))
	assert 0 < opt.fold_i <= opt.n_fold

	is_resume_training = opt.trained_setting_pkl is not None
	if is_resume_training:
		logging.info('Resume training from', opt.trained_setting_pkl)
		opt = np.load(open(opt.trained_setting_pkl, 'rb'), allow_pickle=True).item()
	else:
		opt = post_parse_args(opt)
		setup_running_directories(opt)
		# save the dataset split in setting dictionary
		pprint.pprint(vars(opt))
		# save the setting
		logging.info('Related data will be saved with prefix: %s', opt.exp_prefix)

	assert os.path.exists(opt.input_data_path + "folds/")

	# code which is common for ddi and qm9. take care(P0) if adding other datasets
	data_opt = np.load(open(opt.input_data_path + "input_data.npy",'rb'),allow_pickle=True).item()
	opt.n_atom_type = data_opt.n_atom_type
	opt.n_bond_type = data_opt.n_bond_type
	opt.graph_dict = data_opt.graph_dict
	print(len(opt.graph_dict))
	opt.n_side_effect = None

	if "qm9" in opt.dataset:
		opt.train_graph_dict = pickle.load(open(opt.input_data_path + "folds/" + "train_graphs.npy", "rb"))
		opt.train_labels_dict = pickle.load(open(opt.input_data_path + "folds/" + "train_labels.npy", "rb"))

		opt.valid_graph_dict = pickle.load(open(opt.input_data_path + "folds/" + "valid_graphs.npy", "rb"))
		opt.valid_labels_dict = pickle.load(open(opt.input_data_path + "folds/" + "valid_labels.npy", "rb"))

		# pair train-train, valid-train, test-train
		#TODO start with one copy and compare with multiple
		opt.train_dataset = build_qm9_dataset(graph_dict1=opt.train_graph_dict,
		                                      graph_dict2=opt.train_graph_dict,
		                                      labels_dict1=opt.train_labels_dict,
		                                      labels_dict2=opt.train_labels_dict,
		                                      repetitions=opt.qm9_pairing_repetitions)

		# valid molecule is first in the pair
		opt.valid_dataset = build_qm9_dataset(graph_dict1=opt.valid_graph_dict,
		                                      graph_dict2=opt.train_graph_dict,
		                                      labels_dict1=opt.valid_labels_dict,
		                                      labels_dict2=opt.train_labels_dict,
		                                      repetitions=opt.qm9_pairing_repetitions)

		dataloaders = prepare_qm9_dataloaders(opt)


	if "decagon" in opt.dataset:
		opt.n_side_effect = data_opt.n_side_effect
		opt.side_effects = data_opt.side_effects
		opt.side_effect_idx_dict = data_opt.side_effect_idx_dict

		# 'pos'/'neg' will point to a dictionary where
		# each se points to a list of drug-drug pairs.
		opt.train_dataset = {'pos': {}, 'neg': {}}
		opt.test_dataset = pickle.load(open(opt.input_data_path + "folds/" + str(opt.fold_i) + "fold.npy", "rb"))
		if opt.fold_i == 1:
			valid_fold = 2
		else:
			valid_fold = 1

		opt.valid_dataset = pickle.load(open(opt.input_data_path + "folds/" + str(valid_fold) + "fold.npy", "rb"))

		for i in range(valid_fold+1, opt.n_fold+1):
				if i != opt.fold_i:
					dataset = pickle.load(open(opt.input_data_path + "folds/" + str(i) + "fold.npy", "rb"))
					opt.train_dataset['pos'] = combine(opt.train_dataset['pos'], dataset['pos'])
					opt.train_dataset['neg'] = combine(opt.train_dataset['neg'], dataset['neg'])

		assert data_opt.n_side_effect == len(opt.side_effects)
		dataloaders = prepare_ddi_dataloaders(opt)

	save_experiment_settings(opt)

	# build model
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if torch.cuda.is_available():
		print("using cuda")
	else:
		print("on cpu")

	if opt.transR:
		from model_r import DrugDrugInteractionNetworkR as DrugDrugInteractionNetwork
	elif opt.transH:
		from model_h import DrugDrugInteractionNetworkH as DrugDrugInteractionNetwork
	else:
		from model import DrugDrugInteractionNetwork

	model = DrugDrugInteractionNetwork(
		n_side_effect=opt.n_side_effect,
		n_atom_type=100,
		n_bond_type=20,
		d_node=opt.d_hid,
		d_edge=opt.d_hid,
		d_atom_feat=3,
		d_hid=opt.d_hid,
		d_readout=opt.d_readout,
		n_head=opt.n_attention_head,
		n_prop_step=opt.n_prop_step,
		dropout=opt.dropout,
		score_fn=opt.score_fn).to(device)

	if is_resume_training:
		trained_state = torch.load(opt.best_model_pkl)
		opt.global_step = trained_state['global_step']
		logging.info(
			'Load trained model @ step %d from file: %s',
			opt.global_step, opt.best_model_pkl)
		model.load_state_dict(trained_state['model'])

	train(model, dataloaders, device, opt)


if __name__ == "__main__":
	main()
