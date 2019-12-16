# Can do stuff (e.g. plot histograms) of all_ents
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

sns.set()

entropy = {}
with open("selfrandompair-hid50-readout200-repetitions3-patience8-batch255-mpnn-cv_1_10.pickle", "rb") as f:
	entropy = pickle.load(f)

for i in entropy:
	if i == 0:
		fig = plt.hist(entropy[i], bins=100, alpha=0.5, label="uniform", color='purple')
	else:
		fig = plt.hist(entropy[i], bins=100, alpha=0.5, color='purple')
	plt.xlabel("Entropy")
	plt.ylabel("Number of nodes")
	plt.xlim((-3.5,0))
	plt.subplots_adjust(left=0.15)
	plt.legend()
	plt.savefig(str(i) + "selfrandompair-hid50-readout200-repetitions3-patience8-batch255-mpnn-cv_1_10.png")



