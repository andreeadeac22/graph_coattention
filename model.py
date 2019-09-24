import torch
import torch.nn as nn

import numpy as np

from modules import CoAttentionMessagePassingNetwork
from src.model.model import GraphPairNN


class DrugDrugInteractionNetwork(nn.Module):
	def __init__(
			self,
			n_atom_type, n_bond_type,
			d_node, d_edge, d_atom_feat, d_hid,
			d_readout,
			n_prop_step,
			n_side_effect=None,
			n_lbls = 12,
			n_head=1, dropout=0.1,
			update_method='res', score_fn='trans',
			batch_size=128):

		super().__init__()

		self.dropout = nn.Dropout(p=dropout)

		self.atom_proj = nn.Linear(d_node + d_atom_feat, d_node)
		self.atom_emb = nn.Embedding(n_atom_type, d_node, padding_idx=0)
		self.bond_emb = nn.Embedding(n_bond_type, d_edge, padding_idx=0)
		nn.init.xavier_normal_(self.atom_emb.weight)
		nn.init.xavier_normal_(self.bond_emb.weight)

		self.side_effect_emb = None
		if n_side_effect is not None:
			self.side_effect_emb = nn.Embedding(n_side_effect, d_hid)
			nn.init.xavier_normal_(self.side_effect_emb.weight)

		self.encoder = CoAttentionMessagePassingNetwork(
			d_hid=d_hid, d_readout=d_readout,
			n_head=n_head, n_prop_step=n_prop_step,
			update_method=update_method, dropout=dropout)
		assert update_method == 'res'
		assert score_fn == 'trans'
		self.head_proj = nn.Linear(d_hid, d_hid, bias=False)
		self.tail_proj = nn.Linear(d_hid, d_hid, bias=False)
		nn.init.xavier_normal_(self.head_proj.weight)
		nn.init.xavier_normal_(self.tail_proj.weight)

		self.lbl_predict = nn.Linear(d_readout, n_lbls)

		self.__score_fn = score_fn
		self.pair_model = GraphPairNN(batch_size, batch_size, batch_size).cuda()
		self.batch_size = batch_size

	@property
	def score_fn(self):
		return self.__score_fn

	def forward(
			self,
			seg_m1, atom_type1, atom_feat1, bond_type1,
			inn_seg_i1, inn_idx_j1, out_seg_i1, out_idx_j1,
			seg_m2, atom_type2, atom_feat2, bond_type2,
			inn_seg_i2, inn_idx_j2, out_seg_i2, out_idx_j2,
			se_idx=None, drug_se_seg=None):

		# generate input indices + padding
		ind_in =np.arange(0, self.batch_size)
		ind_in = torch.from_numpy(ind_in).type(torch.FloatTensor)
		ind_in = ind_in.reshape(shape=(1, self.batch_size))

		ind_out = self.pair_model(ind_in.cuda())
		ind_out = ind_out.reshape(shape=(ind_in.shape[0] * ind_in.shape[1], ind_in.shape[1]))
		preds = torch.max(ind_out, dim=1)[1].type(torch.LongTensor)

		atom_batch_dim, n_atom1 = torch.unique(seg_m1, return_counts=True)
		c_atom1 = n_atom1.cumsum(dim=0)-n_atom1[0]
		n_atom1_perm = n_atom1[preds]
		c_atom1_perm = n_atom1_perm.cumsum(dim=0)
		atom_batch_dim, n_atom2 = torch.unique(seg_m2, return_counts=True)
		bond_batch_dim = torch.zeros(inn_seg_i2.shape[0])-1
		bond_batch_dim = bond_batch_dim.type(torch.int).cuda()
		c_atom2 = torch.zeros(n_atom2.shape[0]+1).cuda()
		c_atom2[1:] = n_atom2.cumsum(dim=0)
		c_atom2 = c_atom2.type(torch.LongTensor).cuda()
		n_atom2_perm = n_atom2[preds]
		c_atom2_perm = n_atom2_perm.cumsum(dim=0)
		seg_m2_perm = torch.repeat_interleave(atom_batch_dim, n_atom2_perm)
		atom_shuffle = torch.repeat_interleave(atom_batch_dim * c_atom2_perm, n_atom2_perm)
		cum_c_atom = 0
		for i in range(self.batch_size):
			atom_shuffle[cum_c_atom:cum_c_atom+n_atom2_perm[i]] = c_atom2[preds[i]]+ torch.arange(n_atom2_perm[
																									   i]).cuda()
			cum_c_atom += n_atom2_perm[i]
		atom_type2_perm = atom_type2[atom_shuffle]
		atom_feat2_perm = atom_feat2[atom_shuffle, :]

		# finding batch_dimension of bonds
		for i in range(self.batch_size):
			bond_batch_dim += 1*inn_seg_i2.ge(c_atom2[i]).int().cuda() + 0*inn_seg_i2.lt(c_atom2[i]).int().cuda()


		_ , n_bond2 = torch.unique(bond_batch_dim, return_counts=True)
		c_bond2 = torch.zeros(n_atom2.shape[0] + 1).cuda()
		c_bond2[1:] = n_bond2.cumsum(dim=0)
		c_bond2 = c_bond2.type(torch.LongTensor).cuda()
		bond_seg2 = inn_seg_i2 - torch.repeat_interleave(c_atom2[:-1], n_bond2)
		bond_idx2 = inn_idx_j2 - torch.repeat_interleave(c_atom2[:-1], n_bond2)

		# construct bond shuffle
		n_bond2_perm = n_bond2[preds]
		c_bond2_perm = torch.zeros(n_atom2.shape[0] + 1).cuda()
		c_bond2_perm[1:] = n_bond2_perm.cumsum(dim=0)
		c_bond2_perm = c_bond2_perm.type(torch.LongTensor).cuda()
		bond_shuffle = torch.zeros(c_bond2_perm[128])
		cum_c_bond = 0
		for i in range(self.batch_size):
			bond_shuffle[cum_c_bond:cum_c_bond + n_bond2_perm[i]] = c_bond2[preds[i]] + torch.arange(n_bond2_perm[
																										  i]).cuda()
			cum_c_bond += n_bond2_perm[i]
		bond_shuffle = bond_shuffle.type(torch.LongTensor).cuda()
		bond_type2 = bond_type2[bond_shuffle]
		inn_idx_j2 = bond_idx2[bond_shuffle] + torch.repeat_interleave(c_atom2_perm-c_atom2_perm[0], n_bond2_perm)
		inn_seg_i2 = bond_seg2[bond_shuffle] + torch.repeat_interleave(c_atom2_perm-c_atom2_perm[0], n_bond2_perm)

		# collating drug pairs again
		ddi_seg_i1, ddi_seg_i2, ddi_idx_j1, ddi_idx_j2 = zip(*[
			(i1 + c1, i2 + c2, i2, i1)
			for l1, l2, c1, c2 in zip(n_atom1, n_atom2_perm, c_atom1, c_atom2_perm-n_atom2_perm[0])
			for i1 in range(l1) for i2 in range(l2)])

		seg_m2 = seg_m2_perm.type(torch.LongTensor).cuda()
		atom_type2 = atom_type2_perm.type(torch.LongTensor).cuda()
		atom_feat2 = atom_feat2_perm.type(torch.FloatTensor).cuda()
		bond_type2 = bond_type2.type(torch.LongTensor).cuda()
		inn_seg_i2 = inn_seg_i2.type(torch.LongTensor).cuda()
		inn_idx_j2 = inn_idx_j2.type(torch.LongTensor).cuda()

		out_seg_i1 = torch.LongTensor(ddi_seg_i1).type(torch.LongTensor).cuda()
		out_idx_j1 = torch.LongTensor(ddi_idx_j1).type(torch.LongTensor).cuda()
		out_seg_i2 = torch.LongTensor(ddi_seg_i2).type(torch.LongTensor).cuda()
		out_idx_j2 = torch.LongTensor(ddi_idx_j2).type(torch.LongTensor).cuda()

		atom1 = self.dropout(self.atom_comp(atom_feat1, atom_type1))
		atom2 = self.dropout(self.atom_comp(atom_feat2, atom_type2))

		bond1 = self.dropout(self.bond_emb(bond_type1))
		bond2 = self.dropout(self.bond_emb(bond_type2))

		d1_vec, d2_vec = self.encoder(
			seg_m1, atom1, bond1, inn_seg_i1, inn_idx_j1, out_seg_i1, out_idx_j1,
			seg_m2, atom2, bond2, inn_seg_i2, inn_idx_j2, out_seg_i2, out_idx_j2)

		if self.side_effect_emb is not None:
			d1_vec = d1_vec.index_select(0, drug_se_seg)
			d2_vec = d2_vec.index_select(0, drug_se_seg)

			se_vec = self.dropout(self.side_effect_emb(se_idx))

			fwd_score = self.cal_translation_score(
				head=self.head_proj(d1_vec),
				tail=self.tail_proj(d2_vec),
				rel=se_vec)
			bwd_score = self.cal_translation_score(
				head=self.head_proj(d2_vec),
				tail=self.tail_proj(d1_vec),
				rel=se_vec)
			score = fwd_score + bwd_score

			return score,
		else:
			pred1 = self.lbl_predict(d1_vec)
			pred2 = self.lbl_predict(d2_vec)
			return pred1, pred2, preds


	def atom_comp(self, atom_feat, atom_idx):
		atom_emb = self.atom_emb(atom_idx)
		node = self.atom_proj(torch.cat([atom_emb, atom_feat], -1))
		return node

	def cal_translation_score(self, head, tail, rel):
		return torch.norm(head + rel - tail, dim=1)