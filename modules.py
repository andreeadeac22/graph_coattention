import numpy as np
import torch
import torch.nn as nn


class RadialBasisFunctionExpansion(nn.Module):
	def __init__(self, n_bin, low=0, high=20):
		super().__init__()
		assert high > low

		self.n_center = n_bin
		self.gap = (high - low) / n_bin
		self.centers = nn.Parameter(
			torch.FloatTensor(np.linspace(low, high, n_bin)).view(1, -1))

	def forward(self, dist):
		'''
		input:  dist:   [(b*n_v*n_v) x 1]
		output: rbf:    [(b*n_v*n_v) x n_bin]
		'''

		rbf = dist.unsqueeze(-1) - self.centers  # [(b*n_v*n_v) x n_centers]
		rbf = torch.exp(-(rbf.pow(2)) / self.gap)
		return rbf


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
	def __init__(self, d_in, d_out, n_head=1, dropout=0.1):
		super().__init__()
		self.temperature = np.sqrt(d_in)

		self.n_head = n_head
		self.multi_head = self.n_head > 1

		self.key_proj = nn.Linear(d_in, d_in * n_head, bias=False)
		self.val_proj = nn.Linear(d_in, d_in * n_head, bias=False)
		nn.init.xavier_normal_(self.key_proj.weight)
		nn.init.xavier_normal_(self.val_proj.weight)

		self.attn_drop = nn.Dropout(p=dropout)
		self.out_proj = nn.Sequential(
			nn.Linear(d_in * n_head, d_out),
			nn.LeakyReLU(), nn.Dropout(p=dropout))

	def forward(self, node1, seg_i1, idx_j1, node2, seg_i2, idx_j2, entropy=[]):

		print("node1.shape ", node1.shape)
		print("node2.shape ", node2.shape)

		d_h = node1.size(1)

		n_seg1 = node1.size(0)
		n_seg2 = node2.size(0)

		# Copy center for attention key
		node1_ctr = self.key_proj(node1).index_select(0, seg_i1)
		node2_ctr = self.key_proj(node2).index_select(0, seg_i2)

		# Copy neighbor for attention value
		node1_nbr = self.val_proj(node2).index_select(0, seg_i2)  # idx_j1 == seg_i2
		node2_nbr = self.val_proj(node1).index_select(0, seg_i1)  # idx_j2 == seg_i1

		arg_i1 = None
		arg_i2 = None

		if self.multi_head:
			# prepare copied and shifted index tensors
			seg_i1 = segment_multihead_expand(seg_i1, n_seg1, self.n_head)
			seg_i2 = segment_multihead_expand(seg_i2, n_seg2, self.n_head)

			idx_j1 = idx_j1.unsqueeze(1).expand(-1, self.n_head).contiguous().view(-1)
			idx_j2 = idx_j2.unsqueeze(1).expand(-1, self.n_head).contiguous().view(-1)

			# prepare for the final multi-head concatenation
			arg_i1 = segment_multihead_expand(
				seg_i1.new_tensor(np.arange(n_seg1)), n_seg1, self.n_head)
			arg_i2 = segment_multihead_expand(
				seg_i2.new_tensor(np.arange(n_seg2)), n_seg2, self.n_head)

			# pile up as regular input
			node1_ctr = node1_ctr.view(-1, d_h)
			node2_ctr = node2_ctr.view(-1, d_h)

			node1_nbr = node1_nbr.view(-1, d_h)
			node2_nbr = node2_nbr.view(-1, d_h)

			# new numbers of segments
			n_seg1 = n_seg1 * self.n_head
			n_seg2 = n_seg2 * self.n_head

		translation = (node1_ctr * node2_ctr).sum(1)

		# TODO!! Remove this while training, this is just for entropy.
		#translation = torch.ones_like(translation)

		# Calculate attention weight as edges between two graphs
		node1_edge = self.attn_drop(segment_softmax(
			translation, n_seg1, seg_i1, idx_j1, self.temperature))
		node2_edge = self.attn_drop(segment_softmax(
			translation, n_seg2, seg_i2, idx_j2, self.temperature))

		print("before node1 shape", node1_edge.shape)

		node1_edge = node1_edge.view(-1, 1)
		node2_edge = node2_edge.view(-1, 1)

		print("after node1 shape", node1_edge.shape)


		# Weighted sum
		msg1 = node1.new_zeros((n_seg1, d_h)).index_add(0, seg_i1, node1_edge * node1_nbr)
		msg2 = node2.new_zeros((n_seg2, d_h)).index_add(0, seg_i2, node2_edge * node2_nbr)

		# Entropy computation
		#ent1 = node1.new_zeros((n_seg1, 1)).index_add(0, seg_i1, torch.sum(node1_edge * torch.log(node1_edge + 1e-7), -1))
		#ent2 = node2.new_zeros((n_seg2, 1)).index_add(0, seg_i2, torch.sum(node2_edge * torch.log(node2_edge + 1e-7), -1))
		#entropy.append(ent1)
		#entropy.append(ent2)

		if self.multi_head:
			msg1 = msg1[arg_i1].view(-1, d_h * self.n_head)
			msg2 = msg2[arg_i2].view(-1, d_h * self.n_head)

		msg1 = self.out_proj(msg1)
		msg2 = self.out_proj(msg2)

		return msg1, msg2, node1_edge, node2_edge


class MessagePassing(nn.Module):
	def __init__(self, d_node, d_edge, d_hid, dropout=0.1):
		super().__init__()

		dropout = nn.Dropout(p=dropout)

		self.node_proj = nn.Sequential(
			nn.Linear(d_node, d_hid, bias=False), dropout)

		self.edge_proj = nn.Sequential(
			nn.Linear(d_edge, d_hid), nn.LeakyReLU(), dropout,
			nn.Linear(d_hid, d_hid), nn.LeakyReLU(), dropout)

		self.msg_proj = nn.Sequential(
			nn.Linear(d_hid, d_hid), nn.LeakyReLU(), dropout,
			nn.Linear(d_hid, d_hid), dropout)

	def forward(self, node, edge, seg_i, idx_j):
		edge = self.edge_proj(edge)
		msg = self.node_proj(node)
		msg = self.message_composing(msg, edge, idx_j)
		msg = self.message_aggregation(node, msg, seg_i)
		return msg

	def message_composing(self, msg, edge, idx_j):
		msg = msg.index_select(0, idx_j)  # neighbors
		msg = msg * edge  # element-wise multiplication composition
		return msg

	def message_aggregation(self, node, msg, seg_i):
		msg = torch.zeros_like(node).index_add(0, seg_i, msg)  # sum over messages
		return msg


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
			seg_g1, node1, edge1, inn_seg_i1, inn_idx_j1, out_seg_i1, out_idx_j1,
			seg_g2, node2, edge2, inn_seg_i2, inn_idx_j2, out_seg_i2, out_idx_j2,
						entropies=[]):

		for step_i in range(self.n_prop_step):
			#if step_i >= len(entropies):
			#	entropies.append([])
			inner_msg1 = self.mps[step_i](node1, edge1, inn_seg_i1, inn_idx_j1)
			inner_msg2 = self.mps[step_i](node2, edge2, inn_seg_i2, inn_idx_j2)
			outer_msg1, outer_msg2, attn1, attn2 = self.coats[step_i](
				node1, out_seg_i1, out_idx_j1,
				node2, out_seg_i2, out_idx_j2, [])
				# node2, out_seg_i2, out_idx_j2, entropies[step_i])

			node1 = self.lns[step_i](self.update_fn(node1, inner_msg1, outer_msg1))
			node2 = self.lns[step_i](self.update_fn(node2, inner_msg2, outer_msg2))

		g1_vec = self.readout(node1, seg_g1)
		g2_vec = self.readout(node2, seg_g2)

		return g1_vec, g2_vec, attn1, attn2

	def readout(self, node, seg_g):
		sz_b = seg_g.max() + 1

		node = self.pre_readout_proj(node)
		d_h = node.size(1)

		encv = node.new_zeros((sz_b, d_h)).index_add(0, seg_g, node)
		return encv
