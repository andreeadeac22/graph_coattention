import numpy as np

import torch
from torch.autograd import Variable

def combine(d1, d2):
	for (k, v) in d2.items():
		if k not in d1:
			d1[k] = v
		else:
			d1[k].extend(v)
	return d1


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