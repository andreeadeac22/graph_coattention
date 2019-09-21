import torch
import torch.nn as nn

from modules import CoAttentionMessagePassingNetwork


class DrugDrugInteractionNetwork(nn.Module):
	def __init__(
			self,
			n_side_effect, n_atom_type, n_bond_type,
			d_node, d_edge, d_atom_feat, d_hid,
			n_prop_step, n_head=1, dropout=0.1,
			update_method='res', score_fn='trans'):

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
			d_hid=d_hid, n_head=n_head, n_prop_step=n_prop_step,
			update_method=update_method, dropout=dropout)
		assert update_method == 'res'
		assert score_fn == 'trans'
		self.head_proj = nn.Linear(d_hid, d_hid, bias=False)
		self.tail_proj = nn.Linear(d_hid, d_hid, bias=False)
		nn.init.xavier_normal_(self.head_proj.weight)
		nn.init.xavier_normal_(self.tail_proj.weight)

		self.__score_fn = score_fn

	@property
	def score_fn(self):
		return self.__score_fn

	def forward(
			self,
			seg_m1, atom_type1, atom_feat1, bond_type1,
			inn_seg_i1, inn_idx_j1, out_seg_i1, out_idx_j1,
			seg_m2, atom_type2, atom_feat2, bond_type2,
			inn_seg_i2, inn_idx_j2, out_seg_i2, out_idx_j2,
			se_idx, drug_se_seg):

		atom1 = self.dropout(self.atom_comp(atom_feat1, atom_type1))
		atom2 = self.dropout(self.atom_comp(atom_feat2, atom_type2))

		bond1 = self.dropout(self.bond_emb(bond_type1))
		bond2 = self.dropout(self.bond_emb(bond_type2))

		d1_vec, d2_vec = self.encoder(
			seg_m1, atom1, bond1, inn_seg_i1, inn_idx_j1, out_seg_i1, out_idx_j1,
			seg_m2, atom2, bond2, inn_seg_i2, inn_idx_j2, out_seg_i2, out_idx_j2)

		d1_vec = d1_vec.index_select(0, drug_se_seg)
		d2_vec = d2_vec.index_select(0, drug_se_seg)

		if self.side_effect_emb is not None:
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

	def atom_comp(self, atom_feat, atom_idx):
		atom_emb = self.atom_emb(atom_idx)
		node = self.atom_proj(torch.cat([atom_emb, atom_feat], -1))
		return node

	def cal_translation_score(self, head, tail, rel):
		return torch.norm(head + rel - tail, dim=1)