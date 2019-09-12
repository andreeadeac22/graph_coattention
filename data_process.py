"""
    Data processing.
    #TODO: args, format
"""
import argparse
import json
import networkx as nx
from utils.file_utils import *


def preprocess_decagon(dir_path='./data/'):
    raw_drugs = {}
    with open(dir_path + 'drug_raw_feat.idx.jsonl') as f:
        for l in f:
            idx, l = l.strip().split('\t')
            raw_drugs[idx] = json.loads(l)

    atom_attr_keys = {a_key for d in raw_drugs.values()
                      for a in d['atoms'] for a_key in a.keys()}
    print('Possible atom attribute names:', atom_attr_keys)
    bond_attr_keys = {b_key for d in raw_drugs.values()
                      for b in d['bonds'] for b_key in b.keys()}
    print('Possible bond attribute names:', bond_attr_keys)

    # # Pre-process steps:
    # ## 1. Calculate the number of Hydrogen for every atom.
    # ## 2. Remove Hydrogens in atom list.
    # ## 3. Get final existing bonds.

    def collate_molecule(molecule, self_loop=True):
        atoms = {a['aid']: a for a in molecule['atoms']}

        bonds = {}
        # adding symmetric bonds: (aid1,aid2) as well as (aid2,aid1)
        for b in molecule['bonds']:
            for aid_pair in [(b['aid1'], b['aid2']),
                             (b['aid2'], b['aid1'])]:
                bonds[aid_pair] = '{}-{}'.format(b['order'], b.get('style', 0))

        print("bonds ", bonds)

        if self_loop:
            # Add self loops to the set of existing bonds
            self_bonds = {(aid, aid): 'self' for aid in atoms}
            assert set(self_bonds.keys()) != set(bonds.keys())
            bonds = {**bonds, **self_bonds}

        new_bonds = {}
        # bonds replaces {(b_aid1, b_aid2) : bond_info}
        # with {b_aid1: [(b_aid2, bond_info),...]}
        for aid in atoms:
            atom_vect = []
            for (b_aid1, b_aid2), b in bonds.items():
                if aid == b_aid1:
                    atom_vect.append((b_aid2, b))
            new_bonds[aid] = list(atom_vect)
        bonds = new_bonds

        # Hydrogen bookkeeping
        h_aid_set = {aid for aid, atom in atoms.items() if atom['number'] == 1}

        # {non-hydrogen aid : number of hydrogen bonds it has}
        h_count_dict = {}
        for aid, _ in atoms.items():
            if aid not in h_aid_set:
                hydrogen_neighbour_count = 0
                for nbr, _ in bonds[aid]:
                    if nbr in h_aid_set:
                        hydrogen_neighbour_count += 1
                h_count_dict[aid] = hydrogen_neighbour_count

        assert len(h_aid_set) == sum(h_count_dict.values())
        assert all([0 == a.get('charge', 0) for a in atoms.values() if a['number'] == 1])

        # Remove Hydrogen and use position as new aid
        atoms_wo_h_new_aid = {}
        for idx, (aid, a) in enumerate(
            [(aid, a) for aid, a in atoms.items() if a['number'] > 1]):
            atoms_wo_h_new_aid[aid] = {
                **a,
                'charge': a.get('charge', 0),
                'n_hydro': h_count_dict.get(aid, 0),
                'aid': idx
            }

        # Update with new aid
        bonds_wo_h_new_aid = {}
        for aid1, bs in bonds.items():
            if aid1 not in h_aid_set:
                bonds_wo_h_new_aid[atoms_wo_h_new_aid[aid1]['aid']] =\
                    [(atoms_wo_h_new_aid[aid2]['aid'], b)
                    for aid2, b in bs if aid2 not in h_aid_set]

        atoms_wo_h_new_aid_w_bond = []
        for a in sorted(atoms_wo_h_new_aid.values(), key=lambda x: x['aid']):
            atoms_wo_h_new_aid_w_bond.append(
	            {**a, 'nbr': bonds_wo_h_new_aid[a['aid']]}
            )
        atoms_wo_h_new_aid_w_bond = [  # note the dict key (old aid) is deprecated here.
            {**a, 'nbr': bonds_wo_h_new_aid[a['aid']]}
            for a in sorted(atoms_wo_h_new_aid.values(), key=lambda x: x['aid'])]
        assert all(i == a['aid'] for i, a in enumerate(atoms_wo_h_new_aid_w_bond))

        return atoms_wo_h_new_aid_w_bond

    drug_structure_dict = {cid: collate_molecule(d, self_loop=True)
                           for cid, d in raw_drugs.items()}

    with open(dir_path + 'intermediate_drug_feat.idx.jsonl', 'w') as f:
        for cid, drug in drug_structure_dict.items():
            #drug = drug.to_dict(properties=['atoms', 'bonds'])
            f.write('{}\t{}\n'.format(cid, json.dumps(drug)))

    bond_types = {b for d in drug_structure_dict.values()
                  for a in d for _, b in a['nbr']}
    bond_type_idx = {b: i for i, b in enumerate(bond_types)}

    print('Bond to idx dict:', bond_type_idx)

    def build_graph_idx_mapping(molecule, bond_type_idx=None):
        atom_type = []
        atom_feat = []

        bond_type = []
        bond_seg_i = []
        bond_idx_j = []

        for i, atom in enumerate(molecule):
            print(molecule)
            aid = atom['aid']
            print(i)
            assert aid == i

            nbr_ids, nbr_bonds = zip(*atom['nbr'])
            assert len(set(nbr_ids)) == len(nbr_ids), 'Multi-graph is not supported.'

            if bond_type_idx:
                nbr_bonds = list(map(bond_type_idx.get, nbr_bonds))

            # Follow position i
            atom_feat += [(atom['number'], atom['n_hydro'], atom['charge'])]
            atom_type += [atom['number']]

            # Follow position i
            bond_type += nbr_bonds

            # Follow aid
            bond_seg_i += [aid] * len(nbr_ids)
            bond_idx_j += nbr_ids

        return {'n_atom': len(molecule),
                'atom_type': atom_type,
                'atom_feat': atom_feat,
                'bond_type': bond_type,
                'bond_seg_i': bond_seg_i,
                'bond_idx_j': bond_idx_j}

    drug_graph_dict = {
        cid: build_graph_idx_mapping(d, bond_type_idx=bond_type_idx)
        for cid, d in drug_structure_dict.items()}

    # # Write to jsonl file
    with open(dir_path + 'drug.feat.wo_h.self_loop.idx.jsonl', 'w') as f:
        for cid, d in drug_graph_dict.items():
            f.write('{}\t{}\n'.format(cid, json.dumps(d)))
    with open(dir_path + 'drug.bond_idx.wo_h.self_loop.json', 'w') as f:
        f.write(json.dumps(bond_type_idx))


def preprocess_qm9():
    print("Not implemented")
    return


def main():
    parser = argparse.ArgumentParser(description='Download dataset for Graph Co-attention')
    parser.add_argument('datasets', metavar='D', type=str.lower, nargs='+', choices=['qm9', 'decagon'],
                        help='Name of dataset to download [QM9,DECAGON]')

    # I/O
    parser.add_argument('-p', '--path', metavar='dir', type=str, nargs=1,
                        help="path to store the data (default ./data/)")

    args = parser.parse_args()

    # Check parameters
    if args.path is None:
        args.path = './data/'
    else:
        args.path = args.path[0]

    # Init folder
    prepare_data_dir(args.path)

    if 'qm9' in args.datasets:
        preprocess_qm9(args.path)

    if 'decagon' in args.datasets:
        preprocess_decagon(args.path)


if __name__ == "__main__":
    main()
