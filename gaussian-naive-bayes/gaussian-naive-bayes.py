import numpy as np 

def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    Predict class labels for test samples using Gaussian Naive Bayes.
    """
    # Write code here
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    classes = np.unique(y_train)
    K = len(classes)
    N, D = X_train.shape

    priors = np.array([np.sum(y_train == c) for c in classes]) / N 

    means = np.array([X_train[y_train == c].mean(axis=0) for c in classes])
    vars = np.array([X_train[y_train == c].var(axis=0) for c in classes]) + 1e-9

    log_likelihood = -0.5 * np.sum(np.log(2*np.pi*vars) + \
                                  (X_test[:, None, :] - means)**2 / vars, \
                                  axis = 2)
    log_posterior = np.log(priors) + log_likelihood

    return classes[np.argmax(log_posterior, axis=1)].tolist()
    