import numpy as np
import torch.utils.data


def ddi_collate_paired_batch(paired_batch):
    pos_batch = []
    neg_batch = []
    seg_pos_neg = []
    pos_se_i = 0
    for ddi_pair in paired_batch:
        pos_ddi, neg_ddis = ddi_pair
        pos_batch += [pos_ddi] # flatten negative instances
        neg_batch += neg_ddis
        *_, pos_ses, _ = pos_ddi
        for _ in range(len(pos_ses)):
            seg_pos_neg += [pos_se_i] * len(neg_ddis)
            pos_se_i += 1

    seg_pos_neg = torch.LongTensor(np.array(seg_pos_neg))

    pos_batch = ddi_collate_batch(pos_batch, return_label=False)
    neg_batch = ddi_collate_batch(neg_batch, return_label=False)

    return pos_batch, neg_batch, seg_pos_neg


def ddi_collate_batch(batch, return_label):
    drug1, drug2, se_idx_lists, label = list(zip(*batch))

    ddi_idxs1, ddi_idxs2 = collate_drug_pairs(drug1, drug2)
    drug1 = (*collate_drugs(drug1), *ddi_idxs1)
    drug2 = (*collate_drugs(drug2), *ddi_idxs2)

    se_idx, se_seg = collate_side_effect(se_idx_lists)

    if return_label:
        label = np.hstack([
            [label_i] * len(ses_i) for ses_i, label_i in zip(se_idx_lists, label)])
        return (*drug1, *drug2, se_idx, se_seg, label)
    else:
        return (*drug1, *drug2, se_idx, se_seg)


def collate_drug_pairs(drugs1, drugs2):
    n_atom1 = [d['n_atom'] for d in drugs1]
    n_atom2 = [d['n_atom'] for d in drugs2]
    c_atom1 = [sum(n_atom1[:k]) for k in range(len(n_atom1))]
    c_atom2 = [sum(n_atom2[:k]) for k in range(len(n_atom2))]

    ddi_seg_i1, ddi_seg_i2, ddi_idx_j1, ddi_idx_j2 = zip(*[
        (i1 + c1, i2 + c2, i2, i1)
        for l1, l2, c1, c2 in zip(n_atom1, n_atom2, c_atom1, c_atom2)
        for i1 in range(l1) for i2 in range(l2)])

    ddi_seg_i1 = torch.LongTensor(ddi_seg_i1)
    ddi_idx_j1 = torch.LongTensor(ddi_idx_j1)

    ddi_seg_i2 = torch.LongTensor(ddi_seg_i2)
    ddi_idx_j2 = torch.LongTensor(ddi_idx_j2)

    return (ddi_seg_i1, ddi_idx_j1), (ddi_seg_i2, ddi_idx_j2)


def collate_side_effect(se_idx_lists):
    se_idx = torch.LongTensor(np.hstack(se_idx_lists).astype(np.int64))
    se_seg = np.hstack([[i] * len(ses_i) for i, ses_i in enumerate(se_idx_lists)])
    se_seg = torch.LongTensor(se_seg)
    return se_idx, se_seg


def collate_drugs(drugs):
    c_atoms = [sum(d['n_atom'] for d in drugs[:k]) for k in range(len(drugs))]

    atom_feat = torch.FloatTensor(np.vstack([d['atom_feat'] for d in drugs]))
    atom_type = torch.LongTensor(np.hstack([d['atom_type'] for d in drugs]))
    bond_type = torch.LongTensor(np.hstack([d['bond_type'] for d in drugs]))
    bond_seg_i = torch.LongTensor(np.hstack([
        np.array(d['bond_seg_i']) + c for d, c in zip(drugs, c_atoms)]))
    bond_idx_j = torch.LongTensor(np.hstack([
        np.array(d['bond_idx_j']) + c for d, c in zip(drugs, c_atoms)]))
    batch_seg_m = torch.LongTensor(np.hstack([
        [k] * d['n_atom'] for k, d in enumerate(drugs)]))

    return batch_seg_m, atom_type, atom_feat, bond_type, bond_seg_i, bond_idx_j


class PolypharmacyDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            drug_structure_dict,
            se_idx_dict,
            se_pos_dps=None,
            se_neg_dps=None,
            negative_sampling=False,
            negative_sample_ratio=1,
            n_max_batch_se=1,
            paired_input=False):

        assert se_pos_dps
        assert se_neg_dps or negative_sampling
        assert not (se_neg_dps and negative_sampling)
        assert type(negative_sample_ratio) is int and negative_sample_ratio >= 1

        self.negative_sampling = negative_sampling
        self.paired_input = paired_input

        self.se_idx_dict = se_idx_dict
        """
        print("Se idx dict ")
        with open("se_idx_dict.txt", "w") as filename:
            for se in se_idx_dict:
                print(se, se_idx_dict[se], file=filename)
        """
        self.drug_structure_dict = drug_structure_dict
        """
        print("Drug struct dict ")
        with open("drug_struct_dict.txt", "w") as filename1:
            for drug in drug_structure_dict:
                print(drug, drug_structure_dict[se], file=filename1)
        """
        self.drug_idx_list = list(drug_structure_dict.keys())
        self.n_inst_batch_se = n_max_batch_se
        self.n_corrupt = negative_sample_ratio

        self.pos_ddis = self.collate_given_positive_set(se_pos_dps, se_idx_dict, negative_sampling)
        self.neg_ddis = self.collate_given_negative_set(se_neg_dps, se_idx_dict)

        self.feeding_insts = None
        self.prepare_feeding_insts()

    def collate_given_negative_set(self, se_neg_dps, se_idx_dict):
        ''' From se -> dps mapping to dp -> ses mapping '''
        neg_ddis = {}
        if se_neg_dps:
            for se, dps in se_neg_dps.items():
                for dp in dps:
                    if dp not in neg_ddis:
                        neg_ddis[dp] = []
                    neg_ddis[dp] += [se_idx_dict[se]]
        return neg_ddis

    def mapping_transpose(self, se_dps_dict):
        ''' From `se -> dps` mapping to `dp -> ses` mapping '''
        dp_ses_dict = {}
        for se, dps in se_dps_dict.items():
            for dp in dps:
                if dp not in dp_ses_dict:
                    dp_ses_dict[dp] = []
                dp_ses_dict[dp] += [se]
        return dp_ses_dict

    def collate_given_positive_set(self, se_pos_dps, se_idx_dict, negative_sampling):
        ''' From se -> dps mapping to dp -> ses mapping '''
        pos_ddis = {}
        flip_drug_pair = lambda dp: tuple(reversed(dp))
        for se, dps in se_pos_dps.items():
            if negative_sampling:
                dps = dps + list(map(flip_drug_pair, dps))
            for dp in dps:
                if dp not in pos_ddis:
                    pos_ddis[dp] = []
                pos_ddis[dp] += [se_idx_dict[se]]
        return pos_ddis

    def prepare_feeding_insts(self):

        def collect_with_proper_size_se(ddis, inst_label):
            ''' To reduce the duplicated computing on same graph pair for different labels. '''
            # split number of ses in k * batch(ses) to account
            # for d-d with many vs few ses
            ddis = dict(ddis)
            inst_list = []
            for dp, ses in ddis.items():
                n_se_batch = int(np.ceil(len(ses) / self.n_inst_batch_se))
                for i in range(n_se_batch):
                    start = i * self.n_inst_batch_se
                    end = (i+1) * self.n_inst_batch_se
                    inst_list += [(*dp, ses[start: end], inst_label)]
            return inst_list

        pos_insts = collect_with_proper_size_se(self.pos_ddis, inst_label=True)

        if self.negative_sampling:
            feeding_insts = []
            rand_drugs = list(np.random.choice(
                self.drug_idx_list, size=self.n_corrupt * len(pos_insts)))

            if not self.paired_input:
                feeding_insts = pos_insts

            for pos_inst in pos_insts:
                d1, _, ses, _ = pos_inst
                corr_insts = [(d1, rand_drugs.pop(), ses, False) for _ in range(self.n_corrupt)]

                if self.paired_input:
                    paired_feed = (pos_inst, corr_insts)
                    feeding_insts += [paired_feed]
                else:
                    feeding_insts += corr_insts
        else:
            neg_insts = collect_with_proper_size_se(self.neg_ddis, inst_label=False)
            feeding_insts = pos_insts + neg_insts

        self.feeding_insts = feeding_insts

    def __len__(self):
        return len(self.feeding_insts)

    def __getitem__(self, idx):
        instance = self.feeding_insts[idx]
        # drug lookup
        if self.paired_input:
            pos_inst, neg_insts = instance
            pos_inst = self.drug_structure_lookup(pos_inst)
            neg_insts = list(map(self.drug_structure_lookup, neg_insts))
            return pos_inst, neg_insts
        else:
            instance = self.drug_structure_lookup(instance)
            return instance

    def drug_structure_lookup(self, instance):
        drug_idx1, drug_idx2, se_idx_lists, label = instance
        drug1 = self.drug_structure_dict[drug_idx1]
        drug2 = self.drug_structure_dict[drug_idx2]
        return drug1, drug2, se_idx_lists, label
