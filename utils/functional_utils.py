import random

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
	for i, key1, val1 in enumerate(kv_list1):
		# i-th key in kv_list1
		key2 = kv_list2[i].first
		val2 = kv_list2[i].second

		label1 = labels_dict1[key1]
		label2 = labels_dict2[key2]
		dataset.append(key1,key2,label1,label2)
	return dataset

def build_qm9_dataset(graph_dict1, graph_dict2, labels_dict1, labels_dict2, repetitions):
	datasets = []
	for i in range(repetitions):
		dataset = \
			pair_qm9_graphs(graph_dict1, graph_dict2, labels_dict1, labels_dict2)
		datasets.extend(dataset)
	return datasets