import random
import torch
from torch.autograd import Variable

def combine(d1, d2):
	for (k, v) in d2.items():
		if k not in d1:
			d1[k] = v
		else:
			d1[k].extend(v)
	return d1


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