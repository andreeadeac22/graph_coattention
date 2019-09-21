import numpy as np
import torch.utils.data


def qm9_collate_batch(batch):
	drug1, drug2, label1, label2 = list(zip(*batch))
	ddi_idxs1, ddi_idxs2 = collate_drug_pairs(drug1, drug2)
	drug1 = (*collate_drugs(drug1), *ddi_idxs1)
	drug2 = (*collate_drugs(drug2), *ddi_idxs2)

	labels1 = collate_labels(label1)
	labels2 = collate_labels(label2)
	return (*drug1, *drug2, labels1, labels2)


def collate_drug_pairs(drugs1, drugs2):
	n_atom1 = [d['n_atom'] for d in drugs1]
	n_atom2 = [d['n_atom'] for d in drugs2]
	c_atom1 = [sum(n_atom1[:k]) for k in range(len(n_atom1))]
	c_atom2 = [sum(n_atom2[:k]) for k in range(len(n_atom2))]

	ddi_seg_i1, ddi_seg_i2, ddi_idx_j1, ddi_idx_j2 = zip(*[
		(i1 + c1, i2 + c2, i2, i1)
		for l1, l2, c1, c2 in zip(n_atom1, n_atom2, c_atom1, c_atom2)
		for i1 in range(l1) for i2 in range(l2)])

	ddi_seg_i1 = torch.LongTensor(ddi_seg_i1)
	ddi_idx_j1 = torch.LongTensor(ddi_idx_j1)

	ddi_seg_i2 = torch.LongTensor(ddi_seg_i2)
	ddi_idx_j2 = torch.LongTensor(ddi_idx_j2)

	return (ddi_seg_i1, ddi_idx_j1), (ddi_seg_i2, ddi_idx_j2)


def collate_labels(labels):
	concat_labels = torch.LongTensor(np.hstack(labels).astype(np.int64))
	return concat_labels


def collate_drugs(drugs):
	c_atoms = [sum(d['n_atom'] for d in drugs[:k]) for k in range(len(drugs))]

	atom_feat = torch.FloatTensor(np.vstack([d['atom_feat'] for d in drugs]))
	atom_type = torch.LongTensor(np.hstack([d['atom_type'] for d in drugs]))
	bond_type = torch.LongTensor(np.hstack([d['bond_type'] for d in drugs]))
	bond_seg_i = torch.LongTensor(np.hstack([
		np.array(d['bond_seg_i']) + c for d, c in zip(drugs, c_atoms)]))
	bond_idx_j = torch.LongTensor(np.hstack([
		np.array(d['bond_idx_j']) + c for d, c in zip(drugs, c_atoms)]))
	batch_seg_m = torch.LongTensor(np.hstack([
		[k] * d['n_atom'] for k, d in enumerate(drugs)]))

	return batch_seg_m, atom_type, atom_feat, bond_type, bond_seg_i, bond_idx_j


class QM9Dataset(torch.utils.data.Dataset):
	def __init__(
			self,
			graph_dict,
			pairs_dataset=None):

		print(len(graph_dict))
		assert pairs_dataset
		self.graph_dict = graph_dict
		self.graph_idx_list = list(graph_dict.keys())
		self.feeding_insts = pairs_dataset

	def prepare_feeding_insts(self):
		# Could add optimizing logic here
		return

	def __len__(self):
		return len(self.feeding_insts)

	def __getitem__(self, idx):
		instance = self.feeding_insts[idx]
		# drug lookup
		instance = self.drug_structure_lookup(instance)
		return instance


	def drug_structure_lookup(self, instance):
		drug_idx1, drug_idx2, label1, label2 = instance

		drug1 = self.graph_dict[str(drug_idx1)]
		drug2 = self.graph_dict[str(drug_idx2)]
		return drug1, drug2, label1, label2
