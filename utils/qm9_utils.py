import time
import random
import numpy as np
from tqdm import tqdm
from sklearn import metrics

import torch
import torch.nn as nn

from utils.functional_utils import AverageMeter
from utils.training_data_stats import get_qm9_stats

def scale_labels(label, train_data_min, train_data_max, scale, pref_min=0, pref_max=1):
	scaled_label = label * scale + pref_min - train_data_min * scale
	return scaled_label


def std_labels(std, mean, label):
	standardised_label = (label - mean)/std
	return standardised_label


def build_qm9_dataset(graph_dict1, graph_dict2, labels_dict1, labels_dict2, repetitions, self_pair=False):
	kv_list1 = [(k, v) for k, v in graph_dict1.items()]
	shuffled_lists = {}

	if self_pair:
		kv_list2 = kv_list1
		# repetitions should always be one in this case
		assert repetitions == 1
		for i in range(repetitions):
			shuffled_lists[i] = list(kv_list2)
	else:
		kv_list2 = [(k, v) for k, v in graph_dict2.items()]

		for i in range(repetitions):
			random.shuffle(kv_list2)
			shuffled_lists[i] = list(kv_list2)

	dataset = []

	for i, kv_pair in enumerate(kv_list1):
		# i-th key in kv_list1
		key1 = kv_pair[0]
		val1 = kv_pair[1]
		label1 = labels_dict1[key1]

		for j in range(repetitions):
			kv_list2 = shuffled_lists[j]

			key2 = kv_list2[i][0]
			val2 = kv_list2[i][1]

			label2 = labels_dict2[key2]

			if self_pair:
				assert  key1 == key2

			dataset.append((key1, key2, label1, label2))
	return dataset


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
	debug_loss_fn = nn.L1Loss(reduction='none')

	data_train.dataset.prepare_feeding_insts()
	batch_no = 0

	all_debug_losses = None
	print_step = 200 * opt.qm9_pairing_repetitions

	minima, maxima, mean, std, scale = get_qm9_stats(device)

	for batch in tqdm(data_train, mininterval=3, leave=False, desc='  - (Training)   '):

		# optimize setup
		optimizer.zero_grad()
		update_learning_rate(optimizer, opt.learning_rate, opt.global_step)

		*batch, labels1, labels2 = batch

		batch = [v.to(device) for v in batch]
		labels1 = labels1.to(device)
		labels2 = labels2.to(device)

		# TODO: check if correct
		if opt.qm9_normalise == "scaled":
			labels1 = scale_labels(labels1, minima, maxima, scale)
			labels2 = scale_labels(labels2, minima, maxima, scale)
		else:
			labels1 = std_labels(std, mean, labels1)
			labels2 = std_labels(std, mean, labels2)

		# forward
		pred1, pred2 = model(*batch)

		loss1 = loss_fn(pred1, labels1)
		loss2 = loss_fn(pred2, labels2)
		loss = loss1 + loss2

		if batch_no % print_step == 0:
			debug_loss1 = debug_loss_fn(pred1, labels1)
			debug_loss2 = debug_loss_fn(pred2, labels2)

			if all_debug_losses is None:
				all_debug_losses = torch.cat((
				        debug_loss1.cpu().detach(), debug_loss2.cpu().detach()), 0)
			else:
				all_debug_losses = torch.cat((all_debug_losses,
				            debug_loss1.cpu().detach(), debug_loss2.cpu().detach()), 0)

			print_debug_losses = torch.mean(all_debug_losses, 0)

			print()
			print("At batch_no {0}, loss is {1} and individually \n {2} \n".format(
				batch_no, loss, print_debug_losses.numpy()))

		# backward
		loss.backward()
		optimizer.step()

		# booking
		averaged_model = update_avg_model(model, averaged_model)
		avg_training_loss.update(loss.detach(), 128)
		opt.global_step += 1
		batch_no += 1

	used_time = (time.time() - start) / 60
	return avg_training_loss.get_avg(), used_time, averaged_model


def qm9_valid_epoch(model, data_valid, device, opt, threshold=None):
	model.eval()

	start = time.time()
	loss_fn = nn.L1Loss()
	overall_loss = 0

	debug_loss_fn = nn.L1Loss(reduction='none')
	batch_no = 0

	minima, maxima, mean, std, scale = get_qm9_stats(device)

	all_debug_losses = None

	print_step = 100 * opt.qm9_pairing_repetitions

	with torch.no_grad():
		for batch in tqdm(data_valid, mininterval=3, leave=False, desc='  - (Validation)  '):
			*batch, labels1, labels2 = batch

			batch = [v.to(device) for v in batch]
			labels1 = labels1.to(device)
			labels2 = labels2.to(device)

			# TODO: check if correct
			if opt.qm9_normalise == "scaled":
				labels1 = scale_labels(labels1, minima, maxima, scale)
				labels2 = scale_labels(labels2, minima, maxima, scale)
			else:
				labels1 = std_labels(std, mean, labels1)
				labels2 = std_labels(std, mean, labels2)

			# forward
			pred1, pred2 = model(*batch)

			pred1 = torch.reshape(pred1,
			            (-1, opt.qm9_pairing_repetitions, opt.qm9_output_feat))
			labels1 = torch.reshape(labels1,
			            (-1, opt.qm9_pairing_repetitions, opt.qm9_output_feat))
			assert labels1.shape[2] == opt.qm9_output_feat

			pred1 = torch.mean(pred1, 1)
			labels1 = torch.mean(labels1, 1)

			loss = loss_fn(pred1, labels1)

			debug_loss1 = debug_loss_fn(pred1, labels1)

			# Add just debug_loss1 as debug_loss2 is train
			if all_debug_losses is None:
				all_debug_losses = debug_loss1.cpu().detach()
			else:
				all_debug_losses = torch.cat((all_debug_losses,
				                debug_loss1.cpu().detach()), 0)

			if batch_no % print_step == 0:
				print_debug_losses = torch.mean(all_debug_losses, 0)

				print()
				print("At batch_no {0}, loss is \n {1} \n".format(
					batch_no, print_debug_losses.cpu().detach().numpy()))


			overall_loss += loss.detach()
			batch_no += 1

	overall_losses = torch.mean(all_debug_losses, 0)
	print()
	print("Validation (/test) result ", overall_losses)


	# calculate the performance
	performance = {
		'individual_maes': overall_losses,
		'auroc': torch.mean(overall_losses),
		'threshold': threshold
	}

	used_time = (time.time() - start) / 60
	return performance, used_time


