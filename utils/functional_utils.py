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

	paired_graphs = []
	paired_labels = []
	for i, key1, val1 in enumerate(kv_list1):
		# i-th key in kv_list1

		key2 = kv_list2[i].first
		val2 = kv_list2[i].second

		paired_graphs.append((key1, val1, key2, val2))

		label1 = labels_dict1[key1]
		label2 = labels_dict2[key2]
		paired_labels.append((label1, label2))

	return paired_graphs, paired_labels

def build_qm9_dataset(graph_dict1, graph_dict2, labels_dict1, labels_dict2, repetitions):
	all_graphs = []
	all_labels = []
	for i in range(repetitions):
		paired_graphs, paired_labels = \
			pair_qm9_graphs(graph_dict1, graph_dict2, labels_dict1, labels_dict2)
		all_graphs.extend(paired_graphs)
		all_labels.extend(paired_labels)
	return all_graphs, all_labels