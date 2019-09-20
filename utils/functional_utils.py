def combine(d1, d2):
	for (k, v) in d2.items():
		if k not in d1:
			d1[k] = v
		else:
			d1[k].extend(v)
	return d1