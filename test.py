import pickle
import logging
import argparse
import pprint

import numpy as np
import torch
import torch.utils.data

from dataset import PolypharmacyDataset, collate_batch
from drug_data_util import copy_dataset_from_pkl
from model import DrugDrugInteractionNetwork
from train import valid_epoch as run_evaluation, split_cross_validation_datasets
#, print_performance_table, split_cross_validation_datasets

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S")


def prepare_testset_dataloader(positive_set, negative_set, train_opt, batch_size):
    test_loader = torch.utils.data.DataLoader(
        PolypharmacyDataset(
            drug_structure_dict=train_opt.drug_dict,
            se_idx_dict=train_opt.side_effect_idx_dict,
            se_pos_dps=positive_set,
            se_neg_dps=negative_set,
            n_max_batch_se=1),
        num_workers=2,
        batch_size=batch_size,
        collate_fn=lambda x: collate_batch(x, return_label=True))
    return test_loader


def load_trained_model(train_opt, device):
    if train_opt.transR:
        from model_r import DrugDrugInteractionNetworkR as DrugDrugInteractionNetwork
    elif train_opt.transH:
        from model_h import DrugDrugInteractionNetworkH as DrugDrugInteractionNetwork
    else:
        from model import DrugDrugInteractionNetwork

    model = DrugDrugInteractionNetwork(
        n_side_effect=train_opt.n_side_effect,
        n_atom_type=100,
        n_bond_type=20,
        d_node=train_opt.d_hid,
        d_edge=train_opt.d_hid,
        d_atom_feat=3,
        d_hid=train_opt.d_hid,
        n_head=train_opt.n_attention_head,
        n_prop_step=train_opt.n_prop_step).to(device)
    trained_state = torch.load(train_opt.best_model_pkl)
    model.load_state_dict(trained_state['model'])
    threshold = trained_state['threshold']
    return model, threshold


def main():
    parser = argparse.ArgumentParser()

    # Dirs
    parser.add_argument('setting_pkl')
    parser.add_argument('dataset', metavar='D', type=str.lower,
                        choices=['qm9', 'decagon'],
                        help='Name of dataset to used for training [QM9,DECAGON]')
    parser.add_argument('-t', '--test_dataset_pkl', default=None)
    parser.add_argument('-b', '--batch_size', type=int, default=128)

    eval_opt = parser.parse_args()

    test_opt = np.load(eval_opt.setting_pkl).item()


    if eval_opt.test_dataset_pkl:
        test_dataset = pickle.load(open(eval_opt.test_dataset_pkl, 'rb'))
        positive_data = test_dataset['positive']
        negative_data = test_dataset['negative']
    else:
        """
        train_opt, pos_datasets, neg_datasets = copy_dataset_from_pkl(train_opt)
        positive_data = split_cross_validation_datasets(pos_datasets, train_opt.n_fold, train_opt.fold_i)
        negative_data = split_cross_validation_datasets(neg_datasets, train_opt.n_fold, train_opt.fold_i)
        positive_data = positive_data['test']
        negative_data = negative_data['test']
        """
        test_opt, _, _ = copy_dataset_from_pkl(test_opt)

        with open('cv_data.txt', 'rb') as handle:
            all_cv_datasets = pickle.loads(handle.read())

        positive_data =all_cv_datasets['pos'][test_opt.fold_i]['test']
        negative_data = all_cv_datasets['neg'][test_opt.fold_i]['test']

    # create data loader
    test_data = prepare_testset_dataloader(
    positive_data, negative_data, test_opt, test_opt.batch_size)

    print("batch_size", test_opt.batch_size)

    # build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, threshold = load_trained_model(test_opt, device)

    #print("Threshold", threshold)

    # start testing
    test_perf, _ = run_evaluation(model, test_data, device, test_opt, threshold=threshold)
    for k,v in test_perf.items():
        if k!= 'threshold':
            print(k, v)
    #print_performance_table({k: v for k, v in test_perf.items() if k != 'threshold'})


if __name__ == "__main__":
    main()