import random

def combine(d1, d2):
	for (k, v) in d2.items():
		if k not in d1:
			d1[k] = v
		else:
			d1[k].extend(v)
	return d1


def pair_qm9_graphs(graph_dict, labels_dict):
	kv_list = [(k,v) for k,v in graph_dict.items()]
	permuted_kv_list = list(kv_list)
	random.shuffle(permuted_kv_list)
	assert kv_list != permuted_kv_list

	paired_labels = []
	for i, key, val in enumerate(kv_list):
		label1 = labels_dict[key]
		label2 = labels_dict[permuted_kv_list[i].first]
		paired_labels.append((label1, label2))

	return kv_list, permuted_kv_list, paired_labels

def build_qm9_dataset(graph_dict, labels_dict, repetitions):
	agg =[]
	for i in range(repetitions):
		kv_list, permuted_kv_list, paired_labels = \
			pair_qm9_graphs(graph_dict, labels_dict)

	return agg