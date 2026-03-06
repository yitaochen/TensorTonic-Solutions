import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    """
    Split features X and labels y into train/test while preserving class proportions.
    """
    # Write code here
    # get class counts
    X = np.asarray(X)
    y = np.asarray(y)
    unique_labels = np.unique(y)
    num_classes = len(unique_labels)
    train = []
    test = []
    N = len(X)
    # if rng is None and seed:
    #     rng = np.random.default_rng(seed)
    for unique_label in unique_labels:
        indices = np.where(y == unique_label)[0]
        if rng is not None:
            rng.shuffle(indices)
        else:
            np.random.shuffle(indices)
        # print(indices)
        l = len(indices)
        l_test = round(l * test_size)
        l_train = l - l_test
        # print(l_train)
        if l_train == 0:
            l_train = 1
            l_test = l - 1
        test.extend(indices[:l_test])
        train.extend(indices[l_test:])
    # print(train)
    test.sort()
    train.sort()
    return X[train], X[test], y[train], y[test]
