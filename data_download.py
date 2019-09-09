"""
	File for downloading the training/testing/validation data.
	#TODO: args, output format

"""
import csv
import argparse

def download_decagon_data():
	"""
	Step 0: Download Polypharmacy data
	wget http://snap.stanford.edu/decagon/bio-decagon-combo.tar.gz;
	tar -xvzf bio-decagon-combo.tar.gz;

	Step 1: Collect drug cid list
	"""
	import csv
	drug_idx = set()
	with open('./bio-decagon-combo.csv') as f:
		csv_rdr = csv.reader(f)
		for i, row in enumerate(csv_rdr):
			if i == 0:
				print('Header:', row)
			else:
				drug1, drug2, *_ = row
				drug_idx |= {drug1, drug2}
	print('Instance:', row)

	print('Unique drug count =', len(drug_idx))

	# # Step 2: Search on PubChem

	# In[27]:


	from tqdm import tqdm_notebook
	import pubchempy as pcp
	# Use int type cid to search with PubChemPy
	drugs = {cid: pcp.Compound.from_cid(int(cid.strip('CID')))
	         for cid in tqdm_notebook(drug_idx)}

	# # Step 3: Write to file

	# In[29]:


	import json
	with open('./drug_raw_feat.idx.jsonl', 'w') as f:
		for cid, drug in drugs.items():
			drug = drug.to_dict(properties=['atoms', 'bonds'])
			f.write('{}\t{}\n'.format(cid, json.dumps(drug)))


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default=None, help='Name of dataset to be downloaded')


if __name__ == "__main__":
	main()