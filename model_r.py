import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import CoAttentionMessagePassingNetwork


class DrugDrugInteractionNetworkR(nn.Module):
	def __init__(
			self,
			n_atom_type, n_bond_type,
			d_node, d_edge, d_atom_feat, d_hid,
			n_prop_step,
			n_side_effect=None,
			n_head=1, dropout=0.1,
			update_method='res', score_fn='trans'):

		super().__init__()

		self.dropout = nn.Dropout(p=dropout)

		self.atom_proj = nn.Linear(d_node + d_atom_feat, d_node)
		self.atom_emb = nn.Embedding(n_atom_type, d_node, padding_idx=0)
		self.bond_emb = nn.Embedding(n_bond_type, d_edge, padding_idx=0)
		nn.init.xavier_normal_(self.atom_emb.weight)
		nn.init.xavier_normal_(self.bond_emb.weight)

		self.side_effect_emb = None
		self.se_head_proj_w = None
		self.se_tail_proj_w = None
		if n_side_effect is not None:
			self.side_effect_emb = nn.Embedding(n_side_effect, d_hid)
			nn.init.xavier_normal_(self.side_effect_emb.weight)
			self.se_head_proj_w = nn.Parameter(torch.FloatTensor(n_side_effect, d_hid, d_hid))
			self.se_tail_proj_w = nn.Parameter(torch.FloatTensor(n_side_effect, d_hid, d_hid))
			nn.init.xavier_normal_(self.se_head_proj_w)
			nn.init.xavier_normal_(self.se_tail_proj_w)

		self.encoder = CoAttentionMessagePassingNetwork(
			d_hid=d_hid, n_head=n_head, n_prop_step=n_prop_step,
			update_method=update_method, dropout=dropout)

		if score_fn == 'trans':
			self.head_proj = nn.Linear(d_hid, d_hid, bias=False)
			self.tail_proj = nn.Linear(d_hid, d_hid, bias=False)
			nn.init.xavier_normal_(self.head_proj.weight)
			nn.init.xavier_normal_(self.tail_proj.weight)
			self.scoring_fn = self.cal_sym_translation_score
		elif score_fn == 'factor':
			self.eye = nn.Parameter(torch.eye(d_hid, requires_grad=False).unsqueeze(0))
			self.global_rel = nn.Linear(d_hid, d_hid, bias=False)
			nn.init.xavier_normal_(self.global_rel.weight)
			self.scoring_fn = self.cal_factorize_score
		else:
			raise NotImplementedError

		self.__score_fn = score_fn

	@property
	def score_fn(self):
		return self.__score_fn

	def get_diagonal_matrix(self, vec):
		sz_b, d_vec = vec.size()
		return vec.view(sz_b, d_vec, 1) * self.eye

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

		se_vec = self.dropout(self.side_effect_emb(se_idx))
		se_head_proj = self.dropout(self.se_head_proj_w.index_select(0, se_idx))
		se_tail_proj = self.dropout(self.se_tail_proj_w.index_select(0, se_idx))

		hvecs0 = [se_vec, d1_vec, d2_vec]
		d1_vec = d1_vec.index_select(0, drug_se_seg)
		d2_vec = d2_vec.index_select(0, drug_se_seg)
		#score, hvecs = self.scoring_fn(d1_vec, d2_vec, se_vec, se_head_proj, se_tail_proj)
		fwd_score, hvecs1 = self.cal_translation_score(
			head=d1_vec, tail=d2_vec, rel=se_vec,
			head_proj=se_head_proj, tail_proj=se_tail_proj)
		bwd_score, hvecs2 = self.cal_translation_score(
			head=d2_vec, tail=d1_vec, rel=se_vec,
			head_proj=se_head_proj, tail_proj=se_tail_proj)
		score = fwd_score + bwd_score
		norm_loss = sum([self.cal_vec_norm_loss(v) for v in hvecs0 + hvecs1 + hvecs2])

		return score, norm_loss

	def atom_comp(self, atom_feat, atom_idx):
		atom_emb = self.atom_emb(atom_idx)
		node = self.atom_proj(torch.cat([atom_emb, atom_feat], -1))
		return node

	def cal_vec_norm_loss(self, vec, dim=1):
		norm = torch.norm(vec, dim=dim)
		return torch.mean(F.relu(norm - 1))

	def cal_translation_score(self, head, tail, rel, head_proj, tail_proj):
		proj_head = torch.bmm(head_proj, head.unsqueeze(-1)).view_as(head)
		proj_tail = torch.bmm(tail_proj, tail.unsqueeze(-1)).view_as(tail)
		diff = torch.norm(proj_head + rel - proj_tail, dim=1)
		return diff, [proj_head, proj_tail]


	def cal_sym_translation_score(self, d1_vec, d2_vec, se_vec, head_proj, tail_proj):

		h_d1_vec = torch.bmm(head_proj, d1_vec.unsqueeze(-1)).view_as(d1_vec)
		t_d1_vec = torch.bmm(tail_proj, d1_vec.unsqueeze(-1)).view_as(d1_vec)

		h_d2_vec = torch.bmm(head_proj, d2_vec.unsqueeze(-1)).view_as(d2_vec)
		t_d2_vec = torch.bmm(tail_proj, d2_vec.unsqueeze(-1)).view_as(d2_vec)

		forw_score = torch.norm(h_d1_vec + se_vec - t_d2_vec, dim=1)
		back_score = torch.norm(h_d2_vec + se_vec - t_d1_vec, dim=1)

		translation_score = forw_score + back_score
		return translation_score, [h_d1_vec, t_d1_vec, h_d2_vec, t_d2_vec]
