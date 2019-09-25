import pickle
import logging
import argparse
import random
import pprint
import os

import numpy as np
import torch
import torch.utils.data

from qm9_dataset import QM9Dataset, qm9_collate_batch
from ddi_dataset import PolypharmacyDataset, ddi_collate_batch
#from drug_data_util import copy_dataset_from_pkl
from model import DrugDrugInteractionNetwork
from train import valid_epoch as run_evaluation
from utils.qm9_utils import build_qm9_dataset

logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
	datefmt="%Y-%m-%d %H:%M:%S")

"""
def pair_qm9_test(test_graph_dict, train_graph_dict, test_labels_dict, train_labels_dict):
	train_k_list = list(train_graph_dict.keys())
	test_kv_list = [(k,v) for k,v in test_graph_dict.items()]
	random.shuffle(test_kv_list)
	train_key = random.choice(train_k_list)

	dataset = []
	for i, kv_pair in enumerate(test_kv_list):
		# i-th key in kv_list1
		test_key = kv_pair[0]

		test_lbl = test_labels_dict[test_key]
		train_lbl = train_labels_dict[train_key]
		dataset.append((test_key,train_key,test_lbl,train_lbl))
	return dataset
"""


def prepare_qm9_testset_dataloader(opt):
	test_loader = torch.utils.data.DataLoader(
		QM9Dataset(
			graph_dict=opt.graph_dict,
			pairs_dataset=opt.test_dataset),
		num_workers=2,
		batch_size=opt.batch_size,
		collate_fn=qm9_collate_batch)
	return test_loader


def prepare_ddi_testset_dataloader(positive_set, negative_set, train_opt, batch_size):
	test_loader = torch.utils.data.DataLoader(
		PolypharmacyDataset(
			drug_structure_dict=train_opt.drug_dict,
			se_idx_dict=train_opt.side_effect_idx_dict,
			se_pos_dps=positive_set,
			se_neg_dps=negative_set,
			n_max_batch_se=1),
		num_workers=2,
		batch_size=batch_size,
		collate_fn=lambda x: ddi_collate_batch(x, return_label=True))
	return test_loader


def load_trained_model(train_opt, device):
	if train_opt.transR:
		from model_r import DrugDrugInteractionNetworkR as DrugDrugInteractionNetwork
	elif train_opt.transH:
		from model_h import DrugDrugInteractionNetworkH as DrugDrugInteractionNetwork
	else:
		from model import DrugDrugInteractionNetwork

	model = DrugDrugInteractionNetwork(
		n_side_effect=train_opt.n_side_effect,
		n_atom_type=100,
		n_bond_type=20,
		d_node=train_opt.d_hid,
		d_edge=train_opt.d_hid,
		d_atom_feat=3,
		d_hid=train_opt.d_hid,
		d_readout=train_opt.d_readout,
		n_head=train_opt.n_attention_head,
		n_prop_step=train_opt.n_prop_step).to(device)
	trained_state = torch.load(train_opt.best_model_pkl)
	model.load_state_dict(trained_state['model'])
	threshold = trained_state['threshold']
	return model, threshold


def main():
	parser = argparse.ArgumentParser()

	# Dirs
	parser.add_argument('dataset', metavar='D', type=str.lower,
	                    choices=['qm9', 'decagon'],
	                    help='Name of dataset to used for training [QM9,DECAGON]')
	parser.add_argument('--settings', help='Setting, ends in .npy', default=None)
	parser.add_argument('-mm', '--memo', help='Trained model, ends in .pth', default='default')
	parser.add_argument('--entropy', help='Where to save entropy, ends in .pickle', default=None)
	parser.add_argument('--model_dir', default='./exp_trained')

	parser.add_argument('-t', '--test_dataset_pkl', default=None)
	parser.add_argument('-b', '--batch_size', type=int, default=128)

	eval_opt = parser.parse_args()

	eval_opt.setting_pkl = os.path.join(eval_opt.model_dir, eval_opt.settings)
	eval_opt.best_model_pkl = os.path.join(eval_opt.model_dir, eval_opt.memo)

	test_opt = np.load(eval_opt.setting_pkl, allow_pickle=True).item()
	test_opt.best_model_pkl = eval_opt.best_model_pkl

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if "qm9" in eval_opt.dataset:
		test_opt.test_graph_dict = pickle.load(open(test_opt.input_data_path + "folds/" + "test_graphs.npy", "rb"))
		test_opt.test_labels_dict = pickle.load(open(test_opt.input_data_path + "folds/" + "test_labels.npy", "rb"))

		if not hasattr(test_opt, 'mpnn'):
			test_opt.mpnn = False

		test_opt.test_dataset = build_qm9_dataset(graph_dict1=test_opt.test_graph_dict,
		                                          graph_dict2=test_opt.train_graph_dict,
		                                          labels_dict1=test_opt.test_labels_dict,
		                                          labels_dict2=test_opt.train_labels_dict,
		                                          repetitions=test_opt.qm9_pairing_repetitions,
		                                          self_pair=test_opt.mpnn)

		test_data = prepare_qm9_testset_dataloader(test_opt)

		model, threshold = load_trained_model(test_opt, device)

		test_perf, _ = run_evaluation(model, test_data, device, test_opt)
		for k,v in test_perf.items():
			if k!= 'threshold':
				print(k, v)

		if 'entropy' in test_perf:
			entropy_file_name = os.path.join(eval_opt.model_dir, eval_opt.entropy)
			with open(entropy_file_name, "wb") as ent_file:
				pickle.dump(test_perf['entropy'], ent_file)
	else:
		if eval_opt.test_dataset_pkl:
			test_dataset = pickle.load(open(eval_opt.test_dataset_pkl, 'rb'))
			positive_data = test_dataset['positive']
			negative_data = test_dataset['negative']
		#else:
		#	test_opt, _, _ = copy_dataset_from_pkl(test_opt)

			with open('cv_data.txt', 'rb') as handle:
				all_cv_datasets = pickle.loads(handle.read())

			positive_data =all_cv_datasets['pos'][test_opt.fold_i]['test']
			negative_data = all_cv_datasets['neg'][test_opt.fold_i]['test']

		# create data loader
		test_data = prepare_ddi_testset_dataloader(
			positive_data, negative_data, test_opt, test_opt.batch_size)

		print("batch_size", test_opt.batch_size)

		# build model
		model, threshold = load_trained_model(test_opt, device)

		#print("Threshold", threshold)

		# start testing
		test_perf, _ = run_evaluation(model, test_data, device, test_opt, threshold=threshold)
		for k,v in test_perf.items():
			if k!= 'threshold':
				print(k, v)
		#print_performance_table({k: v for k, v in test_perf.items() if k != 'threshold'})


if __name__ == "__main__":
	main()
