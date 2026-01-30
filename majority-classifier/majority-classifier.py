import numpy as np

def majority_classifier(y_train, X_test):
    """
    Predict the most frequent label in training data for all test samples.
    """
    # Write code here
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    test_shape = X_test.shape 
    unique_vals, counts = np.unique(y_train, return_counts=True)
    maj_idx = np.argmax(counts)
    # print(maj_idx)
    return np.full((test_shape[0], ), unique_vals[maj_idx])