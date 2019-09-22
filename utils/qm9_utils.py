import time
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.functional_utils import AverageMeter, get_optimal_thresholds_for_rels


def pair_qm9_graphs(graph_dict1, graph_dict2, labels_dict1, labels_dict2):
	kv_list1 = [(k,v) for k,v in graph_dict1.items()]
	kv_list2 = [(k,v) for k,v in graph_dict2.items()]
	random.shuffle(kv_list2)
	assert kv_list1 != kv_list2

	dataset = []
	for i, kv_pair in enumerate(kv_list1):
		# i-th key in kv_list1
		key1 = kv_pair[0]
		val1 = kv_pair[1]

		key2 = kv_list2[i][0]
		val2 = kv_list2[i][1]

		label1 = labels_dict1[key1]
		label2 = labels_dict2[key2]
		dataset.append((key1,key2,label1,label2))
	return dataset

def build_qm9_dataset(graph_dict1, graph_dict2, labels_dict1, labels_dict2, repetitions):
	datasets = []
	for i in range(repetitions):
		dataset = \
			pair_qm9_graphs(graph_dict1, graph_dict2, labels_dict1, labels_dict2)
		datasets.extend(dataset)
	return datasets


def qm9_train_epoch(model, data_train, optimizer, averaged_model, device, opt):

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

	# =================================================================================

	model.train()

	start = time.time()
	avg_training_loss = AverageMeter()
	loss_fn = nn.L1Loss()

	data_train.dataset.prepare_feeding_insts()
	for batch in tqdm(data_train, mininterval=3, leave=False, desc='  - (Training)   '):

		# optimize setup
		optimizer.zero_grad()
		update_learning_rate(optimizer, opt.learning_rate, opt.global_step)

		#TODO move to GPU if needed
		#batch = [v.to(device) for v in batch]
		*batch, labels1, labels2 = batch

		# forward
		pred1, pred2 = model(*batch)

		loss1 = loss_fn(pred1, labels1)
		loss2 = loss_fn(pred2, labels2)

		loss = loss1 + loss2

		# backward
		loss.backward()
		optimizer.step()

		# booking
		averaged_model = update_avg_model(model, averaged_model)
		avg_training_loss.update(loss.detach(), 128)
		opt.global_step += 1

	used_time = (time.time() - start) / 60
	return avg_training_loss.get_avg(), used_time, averaged_model


def qm9_valid_epoch(model, data_valid, device, opt, threshold=None):
	model.eval()

	score, label, seidx = [], [], []
	start = time.time()
	with torch.no_grad():
		for batch in tqdm(data_valid, mininterval=3, leave=False, desc='  - (Validation)  '):
			batch = [v.to(device) for v in batch] # move to GPU if needed
			# forward
			predictions, *pos_loss = model(*batch)
			# bookkeeping
			label += [batch_label]
			score += [batch_score]

	cpu = torch.device("cpu")
	label = np.hstack(label)
	score = np.hstack([s.to(cpu) for s in score])

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


