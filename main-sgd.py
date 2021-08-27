import pickle
from random import *
from time import time

import numpy as np
from sklearn.datasets import *
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

seed(100000)

datasets = [load_breast_cancer(), load_digits(), load_iris(), load_wine()]  # , load_linnerud
names = ['Breast_cancer', 'Opt_digits', 'Iris', "Wine"]  # 'load_linnerud'
data_s = [None for i in range(len(datasets))]
target_s = [None for i in range(len(datasets))]
target_names = [None for i in range(len(datasets))]
feature_names = [None for i in range(len(datasets))]
description = [None for i in range(len(datasets))]
for i, dataset in enumerate(datasets):
    data_s[i] = dataset.data
    target_s[i] = dataset.target
    pocket = list(zip(data_s[i], target_s[i]))
    # print(pocket)
    shuffle(pocket)
    data_s[i] = [x[0] for x in pocket]
    target_s[i] = [x[1] for x in pocket]
    feature_names[i] = dataset.feature_names
    description[i] = dataset.DESCR
    target_names[i] = dataset.target_names

# param_Grid = {"max_features":np.linspace(0.001,0.999, num=10), "max_samples":np.linspace(0.001,0.999, num=10), "n_estimators":np.arange(1, 100, 10)}
# param_Grid = {"n_neighbors":np.arange(1,40, 1), "p":np.arange(1,5), "leaf_size":np.arange(1, 31, 1)}
param_Grid = {"learning_rate": ["optimal", 'invscaling', "adaptive", 'constant'], "l1_ratio": np.linspace(0, 1, num=11),
              "alpha": np.logspace(-4, -1, num=40),
              "loss": ["hinge", 'log', 'perceptron', "modified_huber", "squared_hinge"]}
# param_Grid = {"C":np.logspace(-3.5, 4, num=25), "gamma":np.logspace(-5, 2.5, num = 25), "kernel":["rbf", "poly", "sigmoid", "linear"]}
# random_Grid = {"C":loguniform(1e-1, 1e3), "gamma":loguniform(1e-5, 1e0), "kernel":["rbf", "poly", "sigmoid", 'linear']}


x_axis = [2000]
tab = {}
model = SGDClassifier(n_jobs=-1, eta0=0.00001)
for i in range(len(names)):
    tab[names[i]] = [[0 for _ in range(10)] for k in range(10)]
for id in range(10):
    n_itersearch = 200
    """results_rand = {}
    rand_test_score = {}
    rand_train_score = {}
    grid_score = {}
    rand_score = {}
    rand_dict = {}
    time_rand = {}"""

    trainrate = 0.75
    num = 10
    turn = 10
    best_score = [0] * 4
    best_param = [0] * 4
    for k in range(10):
        for i, (name, data) in enumerate(zip(names, datasets)):
            # preprocessing
            print(f"J'ai commencé le traitement du dataset {names[i]}")
            pre_process = StandardScaler()
            x_train, x_test, y_train, y_test = train_test_split(data_s[i], target_s[i], shuffle=False,
                                                                train_size=trainrate)
            x_train, x_test = pre_process.fit_transform(x_train), pre_process.fit_transform(x_test)
            # creation des grilles de recherches structurées et aleatoires
            # grid_t = GridSearchCV(model, param_grid=param_Grid, cv=4, n_jobs=-1, verbose=4)
            rand_t = RandomizedSearchCV(model, param_distributions=param_Grid, n_iter=n_itersearch, cv=3, n_jobs=-1,
                                        verbose=3)
            # tests[names[i]] = [x_train, x_test, y_train, y_test]

            # entrainement et mesure du temps pour le random_search
            start = time()
            rand_t.fit(x_train, y_train)
            if best_score[i] < rand_t.best_score_:
                best_score[i] = rand_t.best_score_
            tab[names[i]][k][id] = best_score[i]

# pd.DataFrame(results_rand).to_csv(f"RANDSEARCH-SVC-{str(total)}")
file_name = "RAN-TAB-SGD"
outfile = open(file_name, "wb")
print(tab)
pickle.dump(tab, outfile)
outfile.close()
