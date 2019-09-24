import sys
import os
import json
import pickle
#sys.path.insert(0, '..')
import molecule_vae
import numpy as np


# 1. load grammar VAE
grammar_weights = "pretrained/zinc_vae_grammar_L56_E100_val.hdf5"
grammar_model = molecule_vae.ZincGrammarModel(grammar_weights)


##################################################################

def extract_graph_properties(prop):
	prop = prop.split()
	g_tag = prop[0]
	g_index = int(prop[1])
	return {"tag": g_tag,
		    "index": g_index}


# XYZ file reader for QM9 dataset
def xyz_graph_reader(graph_file, self_loop=True):
	with open(graph_file, 'r') as f:
		# Number of atoms
		n_atom = int(f.readline())

		# Graph properties
		properties = f.readline()
		prop_dict = extract_graph_properties(properties)
		mol_idx = prop_dict['index']

		atom_properties = []
		# Atoms properties
		for i in range(n_atom):
			a_properties = f.readline()
			a_properties = a_properties.split()
			atom_properties.append(a_properties)

		# Frequencies
		f.readline()

		# SMILES
		smiles = f.readline()
		smiles = smiles.split()
		smiles = smiles[0]
	return mol_idx, smiles


all_smiles = []

dir_path = "../data/qm9/dsgdb9nsd/"


# # Write to pickle file
files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
z1_dict = {}

for file in files:
	if ".xyz" in file:
		mol_idx, smiles = xyz_graph_reader(os.path.join(dir_path, file))
		z1 = grammar_model.encode([smiles])
		z1_dict[mol_idx] = z1

with open(dir_path + 'drug.z1.pickle', 'wb') as f:
	pickle.dump(z1_dict, f)



