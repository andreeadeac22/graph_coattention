import os
import numpy as np

import torch
from torch.autograd import Variable

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from utils.file_utils import *

def combine(d1, d2):
	for (k, v) in d2.items():
		if k not in d1:
			d1[k] = v
		else:
			d1[k].extend(v)
	return d1


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        if isinstance(self.avg, Variable):
            return self.avg.item()
        elif isinstance(self.avg, (torch.Tensor, torch.cuda.FloatTensor)):
            return self.avg.item()
        else:
            raise NotImplementedError


def get_optimal_thresholds_for_rels(relations, gold, score, interval=0.01):

	def get_optimal_threshold(gold, score):
		''' Get the threshold with maximized accuracy'''
		if (np.max(score) - np.min(score)) < interval:
			optimal_threshold = np.max(score)
		else:
			thresholds = np.arange(np.min(score), np.max(score), interval).reshape(1, -1)
			score = score.reshape(-1, 1)
			gold = gold.reshape(-1, 1)
			optimal_threshold_idx = np.sum((score > thresholds) == gold, 0).argmax()
			optimal_threshold = thresholds.reshape(-1)[optimal_threshold_idx]
		return optimal_threshold

	unique_rels = np.unique(relations)
	rel_thresholds = np.zeros(int(unique_rels.max()) + 1)

	for rel_idx in unique_rels:
		rel_mask = np.where(relations == rel_idx)
		rel_gold = gold[rel_mask]
		rel_score = score[rel_mask]
		rel_thresholds[rel_idx] = get_optimal_threshold(rel_gold, rel_score)

	return rel_thresholds


def get_bond_type_idx(bond_type):
	# There are 4 types of bonds in QM9
	# 1 - single, 2 - double, 3 - triple
	# 4 (originally 2.5) - aromatic
	# 0 - self
	if bond_type == 1.5:
		return 4
	return int(bond_type)


def smiles_to_mol(smiles, self_loop, coords_flag=False, atom_properties=[]):
	mol_representation = {
		'n_atom': None,
		'atom_type': None,
		'atom_feat': None,
		'bond_type': None,
		'bond_seg_i': None,
		'bond_idx_j': None,
	}
	m = Chem.MolFromSmiles(smiles)
	m = Chem.AddHs(m)

	fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
	factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
	feats = factory.GetFeaturesForMol(m)

	atom_type = []
	atom_feat = []
	atom_feat_dicts = []

	bond_type = []
	bond_seg_i = []
	bond_idx_j = []
	# distances between atoms
	bond_dist = []

	# Create nodes
	for i in range(0, m.GetNumAtoms()):
		atom_i = m.GetAtomWithIdx(i)

		atom_type += [atom_i.GetAtomicNum()]
		atom_feat += [(atom_i.GetAtomicNum(), atom_i.GetTotalNumHs(), atom_i.GetFormalCharge())]
		atom_feat_dicts += [{
			'number': atom_i.GetAtomicNum(),
			'n_hydro': atom_i.GetTotalNumHs(),
			'charge': atom_i.GetFormalCharge()}]
		if coords_flag:
			atom_feat_dicts[-1]['coord']= np.array(atom_properties[i][1:4]).astype(np.float)


		# TODO: should use the other features too?
		# For example, coordinates, hybridization, aromatic
		# g.add_node(i, a_type=atom_i.GetSymbol(), a_num=atom_i.GetAtomicNum(), acceptor=0, donor=0,
		#		   aromatic=atom_i.GetIsAromatic(), hybridization=atom_i.GetHybridization(),
		#		   num_h=atom_i.GetTotalNumHs(), coord=np.array(atom_properties[i][1:4]).astype(np.float),
		#		   pc=float(atom_properties[i][4]))

		"""
        for i in range(0, len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.node[i]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.node[i]['acceptor'] = 1
        """

	# Read Edges
	for i in range(0, m.GetNumAtoms()):
		for j in range(0, m.GetNumAtoms()):
			e_ij = m.GetBondBetweenAtoms(i, j)
			if e_ij is not None:
				bond_type += [get_bond_type_idx(e_ij.GetBondTypeAsDouble())]
				bond_seg_i += [i]
				bond_idx_j += [j]
				bond_dist += [np.linalg.norm(atom_feat_dicts[i]['coord'] - atom_feat_dicts[j]['coord'])]
		if self_loop:
			# add self edge as type 0
			bond_type += [0]
			bond_seg_i += [i]
			bond_idx_j += [i]
			bond_dist += [0.]

	mol_representation['n_atom'] = m.GetNumAtoms()
	mol_representation['atom_type'] = atom_type
	mol_representation['atom_feat'] = atom_feat
	mol_representation['bond_type'] = bond_type
	mol_representation['bond_seg_i'] = bond_seg_i
	mol_representation['bond_idx_j'] = bond_idx_j

	return mol_representation