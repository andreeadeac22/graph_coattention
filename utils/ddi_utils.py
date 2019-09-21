import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.functional_utils import AverageMeter, get_optimal_thresholds_for_rels

def ddi_train_epoch(model, data_train, optimizer, averaged_model, device, opt):

	def update_avg_model(model, averaged_model):
		decay = 0.9 # moving_avg_decay
		updated_model = model.state_dict()
		for var in updated_model:
			averaged_model[var] = decay * averaged_model[var] + (1 - decay) * updated_model[var]
		return averaged_model

	def update_learning_rate(optimizer, lr_init, global_step):
		lr = lr_init * (0.96 ** (global_step / 1000000))
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

	def max_margin_loss_fn(pos_eg_score, neg_eg_score, seg_pos_neg, margin=1):
		pos_eg_score = pos_eg_score.index_select(0, seg_pos_neg)
		return torch.mean(F.relu(margin - pos_eg_score + neg_eg_score))

	# =================================================================================

	model.train()

	start = time.time()
	avg_training_loss = AverageMeter()

	data_train.dataset.prepare_feeding_insts()
	for batch in tqdm(data_train, mininterval=3, leave=False, desc='  - (Training)   '):

		# optimize setup
		optimizer.zero_grad()
		update_learning_rate(optimizer, opt.learning_rate, opt.global_step)

		if opt.transH:
			model.side_effect_norm_emb.weight = nn.Parameter(
				F.normalize(model.side_effect_norm_emb.weight))

		# move to GPU if needed
		#TODO: qm9 doesn't have positive and negative
		pos_batch, neg_batch, seg_pos_neg = batch
		pos_batch = [v.to(device) for v in pos_batch]
		neg_batch = [v.to(device) for v in neg_batch]
		seg_pos_neg = seg_pos_neg.to(device)

		# forward
		pos_eg_score, *pos_loss = model(*pos_batch)
		neg_eg_score, *neg_loss = model(*neg_batch)

		assert model.score_fn == 'trans'

		#print("len(pos_eg_score)", len(pos_eg_score))

		loss = max_margin_loss_fn(pos_eg_score, neg_eg_score, seg_pos_neg)

		if pos_loss:
			loss += sum(pos_loss) + sum(neg_loss)

		# backward
		loss.backward()
		optimizer.step()

		# booking
		averaged_model = update_avg_model(model, averaged_model)
		sz_b = seg_pos_neg.size(0)
		avg_training_loss.update(loss.detach(), sz_b)
		opt.global_step += 1

	used_time = (time.time() - start) / 60
	return avg_training_loss.get_avg(), used_time, averaged_model


def ddi_valid_epoch(model, data_valid, device, opt, threshold=None):
	model.eval()

	score, label, seidx = [], [], []
	start = time.time()
	with torch.no_grad():
		for batch in tqdm(data_valid, mininterval=3, leave=False, desc='  - (Validation)  '):
			*batch, batch_label = batch
			batch = [v.to(device) for v in batch] # move to GPU if needed
			# forward
			batch_score, *_ = model(*batch)
			# bookkeeping
			label += [batch_label]
			score += [batch_score]
			seidx += [batch[-2]]

	cpu = torch.device("cpu")
	label = np.hstack(label)
	score = np.hstack([s.to(cpu) for s in score])
	seidx = np.hstack([s.to(cpu) for s in seidx])

	if model.score_fn == 'trans':
		''' Unbounded scores'''
		if threshold is None:
			threshold = get_optimal_thresholds_for_rels(seidx, label, score)
		instance_threshold = threshold[seidx]
	else:
		''' logit '''
		def sigmoid(x):
			return 1 / (1 + np.exp(-x))
		score = sigmoid(score) # to prob
		instance_threshold = np.ones_like(score) * 0.5

	pred = score > instance_threshold

	# calculate the performance
	performance = {
		'auroc': metrics.roc_auc_score(label, score),
		'avg_p': metrics.average_precision_score(label, score),
		'f1': metrics.f1_score(label, pred, average='binary'),
		'p': metrics.precision_score(label, pred, average='binary'),
		'r': metrics.recall_score(label, pred, average='binary'),
		'threshold': threshold
	}

	used_time = (time.time() - start) / 60
	return performance, used_time