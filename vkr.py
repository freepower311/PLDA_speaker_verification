from time import time
from scipy.optimize import minimize
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
#import autograd.numpy as np
import numpy as np
#from autograd import grad, value_and_grad
from scipy.io import loadmat
from datetime import datetime
from scipy.special import logsumexp
#from com.eer import calculate_cost, plot_DET_with_EER, find_EER
#import random
from random import shuffle, seed, uniform
seed(0)
CFA=25
CFR=15


def file_time_name():
    return str(datetime.now()).replace(":", "_")[:-7]


old_print = print
log_output = open(f'logs/{file_time_name()}.txt', 'w')


def print(*args, **kwargs):
    old_print(*args, **kwargs)
    old_print(*args, **kwargs, file=log_output)


def load_test_data(file):
    mat = loadmat(file)
    print("load test data:", file)
    print(mat.keys())
    if 'X_female_model' in mat.keys():
        sex = "female"
    else:
        sex = "male"
    X_model = np.array(mat[[*mat][3]])
    X_test = np.array(mat[[*mat][4]])
    Y = np.array(mat[[*mat][5]])

    return X_model, X_test, Y, sex


def load_train_data(file):
    mat = loadmat(file)
    print("load train data:", file)
    print(mat.keys())
    X = np.array(mat[[*mat][4]])
    Y = np.array(mat[[*mat][3]])
    if "X_male_SRE" in mat.keys():
        sex = 'male'
    else:
        sex = 'female'
    return X, Y, sex


def score(theta, D1, D2, All):
    dim = D1.shape[1]
    P = theta[0:dim**2].reshape(dim,dim)
    r = dim**2
    Q = theta[r:r+dim**2].reshape(dim,dim)
    r += dim**2
    c = theta[r:r+dim]
    r += dim
    d = theta[r]
    if All:
        scores = np.dot(D1,np.dot(P,D2.T)) # M * N
        scores += np.sum(np.dot(D1,Q)*D1,1).reshape((D1.shape[0],1)) + np.sum(np.dot(D2,Q)*D2,1).T.reshape((1, D2.shape[0]))
        scores += np.dot(D1,c).reshape((D1.shape[0],1)) + np.dot(D2,c).T.reshape((1, D2.shape[0]))
        scores += d
    else:
        scores = np.sum(np.dot(D1, Q) * D1 + np.dot(D2, Q) * D2 + np.dot(D1, P) * D2, 1) + np.dot(D1 + D2, c) + d
    return scores


def log_loss(theta, X, X_test, Z, lmbda):
    #cost = lambda w: np.sum(np.log(1 + np.exp(score(w, X, X, 1))))
    scores = score(theta, X, X_test, 1)
    scores /= np.max(scores)
    scores = (scores * Z).reshape((Z.shape[0] * Z.shape[1], 1))
    #return np.mean(np.log(1 + np.exp(-scores))) + lmbda * np.dot(theta.T, theta)
    return np.sum(np.log(1 + np.exp(-scores))) + lmbda * np.dot(theta.T, theta)


def restore_theta():
    theta_file = open('theta_male_SRE_logNEW.txt', 'r')
    theta = []
    for line in theta_file:
        theta.append(float(line))
    return np.array(theta)


def restore_far_frr():
    file = open(f'far_log.txt', 'r')
    far = []
    for line in file:
        far.append(float(line))
    file = open(f'frr_log.txt', 'r')
    frr = []
    for line in file:
        frr.append(float(line))
    return far, frr


def calc_metrics(targets_scores, imposter_scores):
    min_score = np.minimum(np.min(targets_scores), np.min(imposter_scores))
    max_score = np.maximum(np.max(targets_scores), np.max(imposter_scores))

    n_tars = len(targets_scores)
    n_imps = len(imposter_scores)

    N = 100

    fars = np.zeros((N,))
    frrs = np.zeros((N,))
    dists = np.zeros((N,))

    mink = float('inf')
    eer = 0

    for i, dist in enumerate(np.linspace(min_score, max_score, N)):
        far = len(np.where(imposter_scores > dist)[0]) / n_imps
        frr = len(np.where(targets_scores < dist)[0]) / n_tars
        fars[i] = far
        frrs[i] = frr
        dists[i] = dist

        k = np.abs(far - frr)

        if k < mink:
            mink = k
            eer = (far + frr) / 2

    return eer, fars, frrs


def sigmoid(x):
    mask = x>=0
    return  mask * 1/(1 + np.exp(-x * mask)) + (1-mask) * np.exp(x*(1-mask)) / (1 + np.exp(x*(1-mask)))


def risk_logistic(theta, X, Z, lmbda):
    N, dim = X.shape
    T = Z * score(theta, X, X, 1)
    mask_pos = T > 0
    mask_neg = T < 0
    Y = mask_pos * np.log(1 + np.exp(-T * mask_pos)) - mask_neg * (T - np.log(1 + np.exp(T * mask_neg)))
    cost = np.mean(Y) + lmbda * np.dot(theta.T, theta)

    G = -Z * sigmoid(-T)
    XG = X.T *np.dot(np.ones((dim, N)), G)
    grad_P = np.dot(X.T,np.dot(G, X))
    grad_Q = 2 * np.dot(XG, X)
    grad_c = 2 * np.sum(XG, 1)
    grad_d = np.sum(G)
    grad = 1 / (N * N) * np.hstack((grad_P.flatten(), grad_Q.flatten(), grad_c.flatten(), grad_d)) + lmbda * theta
    return cost, grad


def risk_hinge( theta, X, Z, lmbda ):
    N, dim = X.shape
    T = Z * score(theta, X, X, 1)
    cost = np.mean(np.maximum(1 - T, 0)) + lmbda / 2 * np.sum(theta**2)
    G = (-Z) * (T < 1)
    XG = X.T * np.dot(np.ones((dim, N)), G)
    grad_P = np.dot(X.T,np.dot(G, X))
    grad_Q = 2 * np.dot(XG, X)
    grad_c = 2 * np.sum(XG, 1)
    grad_d = np.sum(G)
    grad = 1 / (N * N) * np.hstack((grad_P.flatten(), grad_Q.flatten(), grad_c.flatten(), grad_d)) + lmbda * theta
    return cost, grad


def main():
    full_time = time()
    train_files = ["train_data_female_SRE.mat", "train_data_male_SRE.mat"]
    test_files = ["test_data_female_SRE10ext.mat", "test_data_male_SRE10ext.mat"]
    max_iter = 200
    display_minimize = False
    for n_file in [1]:
        print("------------------------------------------------------")

        def get_train_data():
            X, Y, sex = load_train_data(train_files[n_file])
            print("X.shape", X.shape)
            print("Y.shape", Y.shape)
            max_y = np.max(Y)
            print("max_y", max_y)
            return X, Y, sex

        X_full, Y_full, sex = get_train_data()

        def number_classes(Y):
            yy = []
            for y in Y:
                if y not in yy:
                    yy.append(y)
            print("Number of classes", len(yy))

        number_classes(Y)

        dim = X_full.shape[1]
        X_model_orig, X_test_orig, Y1, sex2 = load_test_data(test_files[n_file])

        for train_size in [1000,2000,3000,4000,5000,6000]:
            X = X_full[0:train_size, :]
            Y = Y_full[0:train_size, :]

            t = time()
            Z = np.zeros((X.shape[0], X.shape[0]))
            for i in range(X.shape[0]):
                for j in range(X.shape[0]):
                    if Y[i] == Y[j]:
                        Z[i][j] = 1
                    else:
                        Z[i][j] = -1
            t = time() - t
            print('Z time', t)

            X_model = X_model_orig
            X_test = X_test_orig

            train_size_time = time()
            print("/////////////////")
            print("train_size", train_size)
            print("max_iter", max_iter)
            print("/////////////////")
            methods = [0, 1]
            lmbs = [0.005, 0.01, 0.05]
            for method in methods:
                for lmb in lmbs:
                    if method == 0:
                        method_str = 'logistic'
                        objective = lambda w: risk_logistic(w, X, Z, lmb)
                    else:
                        method_str = 'hinge'
                        objective = lambda w: risk_hinge(w, X, Z, lmb)

                    print("+++++++++++++++++++++++++++++++++++++")
                    print("train_size", train_size)
                    print("loss-function", method_str)
                    print("sex", sex)
                    print("lambda", lmb)

                    w_init = np.random.randn(2 * dim ** 2 + dim + 1)
                    t = time()
                    iter_n = [0]

                    def iter_print(w):
                        iter_n[0] += 1

                    log_theta = minimize(objective, w_init, method='L-BFGS-B',
                                         jac=True,
                                         options={'maxiter': max_iter, 'gtol': 1e-4, 'disp': display_minimize},
                                         callback=iter_print)
                    print("iterations", iter_n[0])
                    theta = log_theta.x
                    file_output = open(f'theta/theta_{sex}_SRE_{method_str}_{train_size}_{lmb}_{file_time_name()}.txt', 'w')
                    t = time() - t
                    print('minimize time', t)
                    for val in theta:
                        file_output.write(str(val) + "\n")
                    file_output.close()
                    sc = score(theta, X_model, X_test, True)
                    print("mean(sc), max(sc), min(sc)", np.mean(sc), np.max(sc), np.min(sc))
                    t = time()

                    scores = sc.flatten()
                    lab = Y1.flatten()
                    target_scores = sorted(scores[lab == 1])
                    impostor_scores = sorted(scores[lab == 0])
                    file_output = open(f'eer/far_frr_{sex}_SRE__{train_size}_{lmb}_{method_str}_{file_time_name()}.txt',
                                       'w')
                    eers, fars, frrs, dists = calc_metrics(target_scores, impostor_scores)
                    print("EER", eers * 100, "%")
                    for val in fars:
                        file_output.write(str(val) + "\n")
                    file_output.write("***\n")
                    for val in frrs:
                        file_output.write(str(val) + "\n")
                    file_output.close()
                    t = time() - t
                    print('eer time', t)
            train_size_time = time() - train_size_time
            print("train_size_time", train_size_time)

    full_time = time() - full_time
    print("full_time", full_time)


if __name__ == '__main__':
    main()
