"""
	Split data for cross-validation and write to file.
"""
import argparse
import numpy as np
import csv
import random

import ujson as json
from tqdm import tqdm

def read_graph_structure(drug_feat_idx_jsonl):
	with open(drug_feat_idx_jsonl) as f:
		drugs = [l.split('\t') for l in f]
		drugs = {idx: json.loads(graph) for idx, graph in tqdm(drugs)}
	return drugs


def prepare_decagon_cv_files(opt):
	def read_ddi_instances(ddi_csv, threshold=498, use_small_dataset=False):
		# Building side-effect dictionary and
		# keeping only those which appear more than threshold (498) times.
		side_effects = {}
		with open(ddi_csv) as csvfile:
			drug_reader = csv.reader(csvfile)
			for i, row in enumerate(drug_reader):
				if i > 0:
					did1, did2, sid, *_ = row
					assert did1 != did2
					if sid not in side_effects:
						side_effects[sid] = []
					side_effects[sid] += [(did1, did2)]

		side_effects = {se: ddis for se, ddis in side_effects.items() if len(ddis) >= threshold}
		if use_small_dataset:  # just for debugging
			side_effects = {se: ddis for se, ddis in
			                sorted(side_effects.items(), key=lambda x: len(x[1]), reverse=True)[:20]}
		print('Total types of polypharmacy side effects =', len(side_effects))
		side_effect_idx_dict = {sid: idx for idx, sid in enumerate(side_effects)}
		return side_effects, side_effect_idx_dict

	def prepare_dataset(se_dps_dict, drug_structure_dict, n_fold=10):
		drug_idx_list = list(drug_structure_dict.keys())
		pos_datasets = {}
		neg_datasets = {}

		for i, se in enumerate(tqdm(se_dps_dict)):
			pos_se_ddp = list(se_dps_dict[se])  # copy
			neg_se_ddp = create_negative_instances(
				drug_idx_list, set(pos_se_ddp), size=len(pos_se_ddp))

			random.shuffle(pos_se_ddp)
			random.shuffle(neg_se_ddp)
			pos_datasets[se] = pos_se_ddp
			neg_datasets[se] = neg_se_ddp

		return pos_datasets, neg_datasets

	def create_negative_instances(drug_idx_list, positive_set, size=None):
		''' For test and validation set'''
		negative_set = set()
		if not size:
			size = len(positive_set)

		while len(negative_set) < size:
			drug1, drug2 = np.random.choice(drug_idx_list, size=2, replace=False)
			assert drug1 != drug2, 'Shall never happen.'

			neg_se_ddp1 = (drug1, drug2)
			neg_se_ddp2 = (drug2, drug1)

			if neg_se_ddp1 in negative_set or neg_se_ddp2 in negative_set:
				continue
			if neg_se_ddp1 in positive_set or neg_se_ddp2 in positive_set:
				continue

			negative_set |= {neg_se_ddp1}
		return list(negative_set)

	# graph_dict is ex drug_dict.
	opt.graph_dict = read_graph_structure(opt.drug_data)
	opt.side_effects, opt.side_effect_idx_dict = read_ddi_instances(
		opt.ddi_data, use_small_dataset=opt.debug)
	opt.pos_datasets, opt.neg_datasets = prepare_dataset(opt.side_effects, opt.drug_dict)
	opt.n_atom_type = 100
	opt.n_bond_type = 20  # 12 in polypharmacy dataset
	opt.n_side_effect = len(opt.side_effects)
	return opt


def prepare_qm9_cv_files(opt):
	opt.graph_dict = read_graph_structure(opt.graph_data)
	opt.n_atom_type = 5 # CHONF
	opt.n_bond_type = 5 # single, double, triple, aromatic, self
	return opt


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('datasets', metavar='D', type=str.lower,
	                    nargs='+', choices=['qm9', 'decagon'],
	                    help='Name of dataset to download [QM9,DECAGON]')

	# I/O
	parser.add_argument('-p', '--path', metavar='dir', type=str, nargs=1,
						help="path to store the data (default ./data/)")

	parser.add_argument('--ddi_data', default='bio-decagon-combo.csv')

	#TODO: modify
	parser.add_argument('--drug_data', default='drug.feat.wo_h.self_loop.idx.jsonl')
	parser.add_argument('-n_fold', default=10, type=int)
	parser.add_argument('--debug', action='store_true')
	opt = parser.parse_args()

	if "qm9" in opt.datasets:
		opt = prepare_qm9_cv_files(opt)
		file_name = opt.path + "qm9/" + "folds/" + opt.n_fold + "fold.npy"

	if "decagon" in opt.datasets:
		opt = prepare_decagon_cv_files(opt)
		file_name = opt.path + "decagon/" + "folds/" + opt.n_fold + "fold.npy"

	print('Dump to file:', file_name)
	np.save(file_name, opt)


if __name__ == '__main__':
	main()

