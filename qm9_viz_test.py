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
from utils.qm9_utils import build_qm9_dataset, build_knn_qm9_dataset

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
			graph_dict=opt.test_graph_dict,
			pairs_dataset=opt.test_dataset),
		num_workers=2,
		batch_size=opt.batch_size,
		collate_fn=qm9_collate_batch)
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
	parser.add_argument('-b', '--batch_size', type=int, default=1)

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

		if not hasattr(test_opt, 'qm9_knn'):
			test_opt.qm9_knn = False

		test_opt.test_dataset = [(1,2,test_opt.test_labels_dict[1], test_opt.test_labels_dict[2])]

		print("test_opt.test_graph_dict[1] ", test_opt.test_graph_dict[1])
		print("test_opt.test_graph_dict[2] ", test_opt.test_graph_dict[2])

		print("test_opt.test_labels_dict[1]" , test_opt.test_labels_dict[1])
		print("test_opt.test_labels_dict[2]" , test_opt.test_labels_dict[2])


		test_data = prepare_qm9_testset_dataloader(test_opt)

		model, threshold = load_trained_model(test_opt, device)

		pred1, pred2, a12, a21 = run_evaluation(model, test_data, device, test_opt)

		viz1 = os.path.join(eval_opt.model_dir, 'viz_attn_coef1.pkl')
		viz2 = os.path.join(eval_opt.model_dir, 'viz_attn_coef2.pkl')


		with open(viz1, 'wb') as h:
			pickle.dump(a12, h)

		with open(viz2, 'wb') as h:
			pickle.dump(a21, h)

		"""
		for k,v in test_perf.items():
			if k!= 'threshold':
				print(k, v)

		if 'entropy' in test_perf:
			entropy_file_name = os.path.join(eval_opt.model_dir, eval_opt.entropy)
			print("entropy_file_name ", entropy_file_name)
			with open(entropy_file_name, "wb") as ent_file:
				pickle.dump(test_perf['entropy'], ent_file)

		#print_performance_table({k: v for k, v in test_perf.items() if k != 'threshold'})
		"""


if __name__ == "__main__":
	main()
