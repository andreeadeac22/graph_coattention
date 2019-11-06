import numpy as np

def split_ids(args, ids, folds=10):

    if args.dataset == 'COLORS-3':
        assert folds == 1, 'this dataset has train, val and test splits'
        train_ids = [np.arange(500)]
        val_ids = [np.arange(500, 3000)]
        test_ids = [np.arange(3000, 10500)]
    elif args.dataset == 'TRIANGLES':
        assert folds == 1, 'this dataset has train, val and test splits'
        train_ids = [np.arange(30000)]
        val_ids = [np.arange(30000, 35000)]
        test_ids = [np.arange(35000, 45000)]
    else:
        n = len(ids)
        stride = int(np.ceil(n / float(folds)))
        test_ids = [ids[i: i + stride] for i in range(0, n, stride)]
        assert np.all(
            np.unique(np.concatenate(test_ids)) == sorted(ids)), 'some graphs are missing in the test sets'
        assert len(test_ids) == folds, 'invalid test sets'
        train_ids = []
        for fold in range(folds):
            train_ids.append(np.array([e for e in ids if e not in test_ids[fold]]))
            assert len(train_ids[fold]) + len(test_ids[fold]) == len(
                np.unique(list(train_ids[fold]) + list(test_ids[fold]))) == n, 'invalid splits'

    return train_ids, test_ids