"""Approximation Vector Machines
"""

from __future__ import division

import time

import numpy as np
import numpy.linalg as LA
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.metrics import mean_squared_error, accuracy_score

# from ...utils import sigmoid

MODE = {"online": 1, "batch": 2}
COVERAGE = {"hypercell": 1, "hypersphere": 2}
LOSS = {"hinge": 1, "l1": 2, "l2": 3, "logit": 4, "eps_insensitive": 5}
TASK = {"classification": 1, "regression": 2}
KERNEL = {"gaussian": 1, "linear": 2}
DISTANCE = {"euclidean": 1, "hamming": 2, "cosine": 3}

EPS = np.finfo(np.float).eps


class AVM(BaseEstimator, ClassifierMixin, RegressorMixin):

    def __init__(self, cover="hypercell", delta=0.1, dist="euclidean",
                 loss="hinge", eps=0.1, mode="online",
                 kernel="gaussian", gamma=0.1,
                 lbd=1.0, avg_weight=False, beta=0, rho=1.0, max_size=1000,
                 record=-1, verbose=0):
        self.cover = cover
        self.delta = delta
        self.dist = dist
        self.loss = loss
        self.eps = eps
        self.mode = mode
        self.kernel = kernel
        self.gamma = gamma
        self.lbd = lbd
        self.avg_weight = avg_weight
        self.beta = beta
        self.rho = rho
        self.max_size = max_size
        self.record = record
        self.verbose = verbose

    def init(self):
        try:
            self.cover = COVERAGE[self.cover]
        except KeyError:
            raise ValueError("Cover type %s is not supported." % self.cover)

        try:
            self.dist = DISTANCE[self.dist]
        except KeyError:
            raise ValueError("Distance type %s is not supported." % self.dist)

        try:
            self.loss = LOSS[self.loss]
        except KeyError:
            raise ValueError("Loss function %s is not supported." % self.loss)

        try:
            self.mode = MODE[self.mode]
        except KeyError:
            raise ValueError("Learning mode % is not supported." % self.mode)

        try:
            self.kernel = KERNEL[self.kernel]
        except KeyError:
            raise ValueError("Kernel %s is not supported." % self.kernel)

        if self.loss == LOSS["hinge"] or self.loss == LOSS["logit"]:
            self.task_ = TASK["classification"]
        else:
            self.task_ = TASK["regression"]

        self.n_classes_ = 0
        self.class_name_ = None
        self.size_ = 0
        self.w_ = None
        self.X_core_ = None
        self.X_dict_ = {}
        self.train_time_ = []
        self.last_train_time_ = 0
        self.score_ = []
        self.score_idx_ = []
        self.last_score_ = 0

    def get_idx(self, x):
        if self.size_ == 0:
            return -1
        else:
            if self.dist == DISTANCE["euclidean"]:
                t = (self.X_core_[:self.size_]-x)
                d = np.sqrt(np.sum(t*t, axis=1))
            elif self.dist == DISTANCE["hamming"]:
                t = (self.X_core_[:self.size_].astype(np.uint8) ^ x.astype(np.uint8))
                d = np.sum(t, axis=1)
            elif self.dist == DISTANCE["cosine"]:
                X_norm = LA.norm(self.X_core_[:self.size_], axis=1)
                Y_norm = LA.norm(x)
                d = 1 - np.sum(x*self.X_core_[:self.size_], axis=1) / (EPS + Y_norm*X_norm)

            if self.cover == COVERAGE["hypercell"]:
                for i in np.argsort(d):
                    lower_bound = self.X_core_[i] - self.delta
                    upper_bound = self.X_core_[i] + self.delta
                    if np.all(np.uint8(x >= lower_bound) & np.uint8(x <= upper_bound)):
                        return i
                return -1
            elif self.cover == COVERAGE["hypersphere"]:
                idx = np.argmin(d)
                return idx if d[idx] <= self.delta else -1
            else:
                return -1

    def add_to_core_set(self, x, w, k=None, y=None, z=None):
        self.X_core_[self.size_] = x
        if y is not None:
            self.w_[self.size_, y] = -w
            if z >= 0:
                self.w_[self.size_, z] = w
        else:
            self.w_[self.size_] = w
        if k is not None:
            self.X_dict_[k] = self.size_
        self.size_ += 1

    def get_wxy(self, x, y, wx=None):
        if self.size_ == 0:
            return (0, -1)
        else:
            if self.kernel == KERNEL["gaussian"]:
                if wx is None:
                    t = (self.X_core_[:self.size_]-x)
                    wx = np.sum(self.w_[:self.size_, :]*np.exp(-self.gamma*np.sum(t*t, axis=1, keepdims=True)), axis=0)
                idx = np.ones(self.n_classes_, np.bool)
                idx[y] = False
                z = np.argmax(wx[idx])
                z += (z >= y)
                return (wx[y] - wx[z], z)
            else:
                return (0, -1)

    def get_wx(self, x):
        if self.size_ == 0:
            return [0]
        else:
            if self.kernel == KERNEL["gaussian"]:
                t = (self.X_core_[:self.size_]-x)
                return np.sum(self.w_[:self.size_]*np.exp(-self.gamma*np.sum(t*t, axis=1, keepdims=True)), axis=0)
            elif self.kernel == KERNEL["linear"]:
                t = self.X_core_[:self.size_]*x
                return np.sum(self.w_[:self.size_]*np.sum(t, axis=1, keepdims=True), axis=0)
            else:
                return [0]

    def get_grad(self, x, y, wx=None):
        if self.n_classes_ > 2:
            wxy, z = self.get_wxy(x, y, wx)
            if self.loss == LOSS["hinge"]:
                return (-1, z) if wxy <= 1 else (0, z)
            else:
                if wxy > 0:
                    return (-np.exp(-wxy) / (np.exp(-wxy) + 1), z)
                else:
                    return (-1 / (1 + np.exp(wxy)), z)
        else:
            if wx is None:
                wx = self.get_wx(x)[0]
            else:
                wx = wx[0]
            if self.loss == LOSS["hinge"]:
                return (-y, -1) if y*wx <= 1 else (0, -1)
            elif self.loss == LOSS["l1"]:
                return (np.sign(wx - y), -1)
            elif self.loss == LOSS["l2"]:
                return (wx-y, -1)
            elif self.loss == LOSS["logit"]:
                if y*wx > 0:
                    return (-y*np.exp(-y*wx) / (np.exp(-y*wx) + 1), -1)
                else:
                    return (-y / (1 + np.exp(y*wx)), -1)
            elif self.loss == LOSS["eps_insensitive"]:
                return (np.sign(wx - y), -1) if np.abs(y - wx) > self.eps else (0, -1)

    def fit(self, X, y):
        self.init()

        if self.mode == MODE["online"]:
            y0 = y

        if self.task_ == TASK["classification"]:
            self.class_name_, y = np.unique(y, return_inverse=True)
            self.n_classes_ = len(self.class_name_)
            if self.n_classes_ == 2:
                y[y == 0] = -1

        if self.n_classes_ > 2:
            self.w_ = np.zeros([self.max_size, self.n_classes_])
        else:
            self.w_ = np.zeros([self.max_size, 1])

        if self.avg_weight:
            w_avg = np.zeros(self.w_.shape)

        self.X_core_ = np.zeros((self.max_size, X.shape[1]))

        score = 0.0
        start_time = time.time()
        for t in xrange(X.shape[0]):

            if self.mode == MODE["online"]:
                wx = self.get_wx(X[t])
                if self.task_ == TASK["classification"]:
                    if self.n_classes_ == 2:
                        y_pred = self.class_name_[wx[0] >= 0]
                    else:
                        y_pred = self.class_name_[np.argmax(wx)]
                    score += (y_pred != y0[t])
                else:
                    score += (wx[0]-y0[t])**2
                alpha_t, z = self.get_grad(X[t], y[t], wx=wx)   # compute \alpha_t
            else:
                alpha_t, z = self.get_grad(X[t], y[t])     # compute \alpha_t

            self.w_ *= 1.0*t/(t+1)

            is_approx = np.random.rand() <= max(0, 1-self.beta/((t+1)*self.rho))
            idx = self.get_idx(X[t]) if is_approx else -1
            if idx > -1:
                if self.n_classes_ > 2:
                    self.w_[idx, y[t]] -= alpha_t/(self.lbd*(t+1))
                    if z >= 0:
                        self.w_[idx, z] += alpha_t/(self.lbd*(t+1))
                else:
                    self.w_[idx] -= alpha_t/(self.lbd*(t+1))
            elif self.size_ < self.max_size:
                if self.n_classes_ > 2:
                    self.add_to_core_set(X[t], alpha_t/(self.lbd*(t+1)), y=y[t], z=z)
                else:
                    self.add_to_core_set(X[t], -alpha_t/(self.lbd*(t+1)))
            # else:
            #     print "[WARN] Maximum core set size has been reached."

            if self.avg_weight:
                w_avg += self.w_

            if self.record > 0 and (not ((t+1) % self.record)):
                self.train_time_.append(time.time()-start_time)
                self.score_.append(score/(t+1))
                self.score_idx_.append(t+1)

            if self.verbose:
                print "[INFO] Data point: %d\tModel size: %d\tElapsed time: %.4f" % (t, self.size_, time.time()-start_time)

        if self.avg_weight:
            self.w_ = w_avg / X.shape[0]

        if self.mode == MODE["online"]:
            self.last_train_time_ = time.time() - start_time
            self.last_score_ = score / X.shape[0]
        else:
            self.train_time_ = time.time() - start_time

        return self

    def predict(self, X):
        y = np.zeros(X.shape[0])
        for i in xrange(X.shape[0]):
            wx = self.get_wx(X[i])
            if self.task_ == TASK["classification"]:
                if self.n_classes_ == 2:
                    y[i] = self.class_name_[wx[0] >= 0]
                else:
                    y[i] = self.class_name_[np.argmax(wx)]
            else:
                y[i] = wx[0]
        return y

    def score(self, X, y):
        if self.mode == MODE["online"]:
            return -self.last_score_
        else:
            if self.task_ == TASK["classification"]:
                return float(accuracy_score(self.predict(X), y))
            else:
                return -float(mean_squared_error(self.predict(X), y))


class OnlineMulticlassAVM(BaseEstimator, ClassifierMixin, RegressorMixin):

    def __init__(self, delta=0.1, loss="hinge", gamma=0.1, lbd=1.0, max_size=1000, record=-1):
        self.delta = delta
        self.loss = loss
        self.gamma = gamma
        self.lbd = lbd
        self.max_size = max_size
        self.record = record

    def init(self):
        try:
            self.loss = LOSS[self.loss]
        except KeyError:
            raise ValueError("Loss function %s is not supported." % self.loss)

        self.n_classes_ = 0
        self.class_name_ = None
        self.size_ = 0
        self.w_ = None
        self.X_core_ = None
        self.train_time_ = []
        self.last_train_time_ = 0
        self.score_ = []
        self.score_idx_ = []
        self.last_score_ = 0

    def fit(self, X, y):
        self.init()

        y0 = y

        self.class_name_, y = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.class_name_)

        self.w_ = np.zeros((self.max_size, self.n_classes_))

        self.X_core_ = np.zeros((self.max_size, X.shape[1]))

        score = 0.0
        mask = np.ones(self.n_classes_, np.bool)
        start_time = time.time()
        for t in xrange(X.shape[0]):

            z = -1
            if self.size_ > 0:
                d = (self.X_core_[:self.size_]-X[t])
                d2 = np.sum(d*d, axis=1, keepdims=True)
                wx = np.sum(self.w_[:self.size_]*np.exp(-self.gamma*d2), axis=0)
                y_pred = self.class_name_[np.argmax(wx)]
                score += (y_pred != y0[t])

                # wxy, z = self.get_wxy(X[t], y[t], wx)
                mask[y[t]] = False
                z = np.argmax(wx[mask])
                mask[y[t]] = True
                z += (z >= y[t])
                wxy = wx[y[t]] - wx[z]

                if self.loss == LOSS["hinge"]:
                    alpha_t = -1 if wxy <= 1 else 0
                else:
                    if wxy > 0:
                        alpha_t = -np.exp(-wxy) / (np.exp(-wxy) + 1)
                    else:
                        alpha_t = -1 / (1 + np.exp(wxy))
            else:
                alpha_t = 0

            self.w_ *= 1.0*t/(t+1)

            if self.size_ > 0:
                # idx = self.get_idx(X[t])
                idx = np.argmin(d2.reshape(-1))
                if d2[idx] > self.delta*self.delta:
                    idx = -1
            else:
                idx = -1
            if idx > -1:
                self.w_[idx, y[t]] -= alpha_t/(self.lbd*(t+1))
                if z >= 0:
                    self.w_[idx, z] += alpha_t/(self.lbd*(t+1))
            elif self.size_ < self.max_size:
                # self.add_to_core_set(X[t], alpha_t/(self.lbd*(t+1)), y=y[t], z=z)
                self.X_core_[self.size_] = X[t]
                self.w_[self.size_, y[t]] = -alpha_t/(self.lbd*(t+1))
                if z >= 0:
                    self.w_[self.size_, z] = alpha_t/(self.lbd*(t+1))
                self.size_ += 1
            else:
                print "[WARN] Maximum core set size has been reached."

            if self.record > 0 and (not ((t+1) % self.record)):
                self.train_time_.append(time.time()-start_time)
                self.score_.append(score/(t+1))
                self.score_idx_.append(t+1)

        self.last_train_time_ = time.time() - start_time
        self.last_score_ = score / X.shape[0]

        return self
