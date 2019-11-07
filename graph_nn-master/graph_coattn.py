import numpy as np
import torch
import torch.nn as nn


def segment_max(logit, n_seg, seg_i, idx_j):
	max_seg_numel = idx_j.max().item() + 1
	seg_max = logit.new_full((n_seg, max_seg_numel), -np.inf)
	seg_max = seg_max.index_put_((seg_i, idx_j), logit).max(dim=1)[0]
	return seg_max[seg_i]


def segment_sum(logit, n_seg, seg_i):
	norm = logit.new_zeros(n_seg).index_add(0, seg_i, logit)
	return norm[seg_i]


def segment_softmax(logit, n_seg, seg_i, idx_j, temperature):
	logit_max = segment_max(logit, n_seg, seg_i, idx_j).detach()
	logit = torch.exp((logit - logit_max) / temperature)
	logit_norm = segment_sum(logit, n_seg, seg_i)
	prob = logit / (logit_norm + 1e-8)
	return prob


def segment_multihead_expand(seg_i, n_seg, n_head):
	i_head_shift = n_seg * seg_i.new_tensor(torch.arange(n_head))
	seg_i = (seg_i.view(-1, 1) + i_head_shift.view(1, -1)).view(-1)
	return seg_i


class CoAttention(nn.Module):
	def __init__(self,
			d_in,
			d_out,
			n_head=1,
			dropout=0.1):
		super().__init__()
		self.temperature = np.sqrt(d_in)

		self.n_head = n_head
		self.multi_head = self.n_head > 1

		self.key_proj = nn.Linear(d_in, d_in * n_head, bias=False)
		self.val_proj = nn.Linear(d_in, d_in * n_head, bias=False)
		nn.init.xavier_normal_(self.key_proj.weight)
		nn.init.xavier_normal_(self.val_proj.weight)

		self.softmax = nn.Softmax(dim=1)

		self.attn_drop = nn.Dropout(p=dropout)
		self.out_proj = nn.Sequential(
			nn.Linear(d_in * n_head, d_out),
			nn.LeakyReLU(), nn.Dropout(p=dropout))


	def forward(self, x1, x2, entropy=[]):

		# Why is there a key_proj and a val_proj?
		# Copy center for attention key
		#node1_ctr = self.key_proj(node1).index_select(0, seg_i1)
		#node2_ctr = self.key_proj(node2).index_select(0, seg_i2)
		# Copy neighbor for attention value
		#node1_nbr = self.val_proj(node2).index_select(0, seg_i2)  # idx_j1 == seg_i2
		#node2_nbr = self.val_proj(node1).index_select(0, seg_i1)  # idx_j2 == seg_i1

		h1 = self.key_proj(x1)
		h2 = self.key_proj(x2)

		e12 = torch.bmm(h1, torch.transpose(h2, -1, -2))
		e21 = torch.bmm(h2, torch.transpose(h1, -1, -2))

		# what is segment_softmax
		# dropout?
		a12 = self.softmax(e12)
		a21 = self.softmax(e21)

		n12 = torch.bmm(a12, self.val_proj(x2))
		n21 = torch.bmm(a21, self.val_proj(x1))

		# TODO!! Remove this while training, this is just for entropy.
		#translation = torch.ones_like(translation)

		# Entropy computation
		#ent1 = node1.new_zeros((n_seg1, 1)).index_add(0, seg_i1, torch.sum(node1_edge * torch.log(node1_edge + 1e-7), -1))
		#ent2 = node2.new_zeros((n_seg2, 1)).index_add(0, seg_i2, torch.sum(node2_edge * torch.log(node2_edge + 1e-7), -1))
		#entropy.append(ent1)
		#entropy.append(ent2)

		msg1 = self.out_proj(n12)
		msg2 = self.out_proj(n21)

		#Check if 1 -> 1 and 2 -> 2.
		return msg1, msg2


class MessagePassing(nn.Module):
	def __init__(self,
				 node_in_feat, # probably d_node
				 node_out_feat, # probably d_hid
				 dropout = 0.1):
		super(MessagePassing, self).__init__()

		dropout = nn.Dropout(p=dropout)
		self.node_proj = nn.Sequential(
			nn.Linear(node_in_feat, node_out_feat, bias=False))
			#, dropout)

	def compute_adj_mat(self, A):
		batch, N = A.shape[:2]
		A_hat = A
		I = torch.eye(N).unsqueeze(0).to(cuda_device)
		A_hat = A + I
		return A_hat

	def forward(self, x, A, mask):
		# print('in', x.shape, torch.sum(torch.abs(torch.sum(x, 2)) > 0))
		if len(A.shape) == 3:
			A = A.unsqueeze(3)

		new_A = self.compute_adj_mat(A[:, :, :, 0]) #only one relation type
		x = self.fc(x)
		x = torch.bmm(new_A, x)

		if len(mask.shape) == 2:
			mask = mask.unsqueeze(2)
		x = x * mask  # to make values of dummy nodes zeros again, otherwise the bias is added after applying self.fc which affects node embeddings in the following layers
		return (x, A, mask)


class CoAttentionMessagePassingNetwork(nn.Module):
	def __init__(self, d_hid, d_readout, n_prop_step, n_head=1, dropout=0.1, update_method='res'):

		super().__init__()

		self.n_prop_step = n_prop_step

		if update_method == 'res':
			x_d_node = lambda step_i: 1
			self.update_fn = lambda x, y, z: x + y + z
		elif update_method == 'den':
			x_d_node = lambda step_i: 1 + 2 * step_i
			self.update_fn = lambda x, y, z: torch.cat([x, y, z], -1)
		else:
			raise NotImplementedError

		self.mps = nn.ModuleList([
			MessagePassing(
				d_node=d_hid * x_d_node(step_i),
				d_edge=d_hid, d_hid=d_hid, dropout=dropout)
			for step_i in range(n_prop_step)])

		self.coats = nn.ModuleList([
			CoAttention(
				d_in=d_hid * x_d_node(step_i),
				d_out=d_hid, n_head=n_head, dropout=dropout)
			for step_i in range(n_prop_step)])

		self.lns = nn.ModuleList([
			nn.LayerNorm(d_hid * x_d_node(step_i))
			for step_i in range(n_prop_step)])

		self.pre_readout_proj = nn.Sequential(
			nn.Linear(d_hid * x_d_node(n_prop_step), d_readout),
			nn.LeakyReLU())

	def forward(
			self,
			data,
			entropies=[]):

		x, A, masks = data[:3]

		evens = torch.tensor([i for i in range(0,x.shape[0],2)])
		odds = torch.tensor([i for i in range(1,x.shape[0],2)])

		x1 = torch.index_select(x, 0, evens)
		A1 = torch.index_select(A, 0, evens)
		masks1 = torch.index_select(masks, 0, evens)

		x2 = torch.index_select(x, 0, odds)
		A2 = torch.index_select(A, 0, odds)
		masks2 = torch.index_select(masks, 0, odds)

		for step_i in range(self.n_prop_step):
			#if step_i >= len(entropies):
			#	entropies.append([])
			inner_msg1 = self.mps[step_i](x1, A1, masks1)
			inner_msg2 = self.mps[step_i](x2, A2, masks2)
			outer_msg1, outer_msg2 = self.coats[step_i](
				x1,
				x2, [])
				# node2, out_seg_i2, out_idx_j2, entropies[step_i])

			node1 = self.lns[step_i](self.update_fn(x1, inner_msg1, outer_msg1))
			node2 = self.lns[step_i](self.update_fn(x2, inner_msg2, outer_msg2))

		g1_vec = self.readout(node1, seg_g1)
		g2_vec = self.readout(node2, seg_g2)

		return g1_vec, g2_vec

	def readout(self, node, seg_g):
		sz_b = seg_g.max() + 1

		node = self.pre_readout_proj(node)
		d_h = node.size(1)

		encv = node.new_zeros((sz_b, d_h)).index_add(0, seg_g, node)
		return encv
