import sklearn
from random import *
from matplotlib import  pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from numpy.linalg import *
from sklearn.datasets import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, validation_curve, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
import seaborn as sns
import pandas as pd
from time import time
import scipy.stats as stats
from sklearn.utils.fixes import loguniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import SGDClassifier
import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from statistics import *
from deap import cma

random.seed(100000)

datasets = [load_breast_cancer(), load_digits(), load_iris(), load_wine()]#, load_linnerud
names = ['Breast_cancer', 'Opt_digits', 'Iris', "Wine"]# 'load_linnerud'
data_s = [None for i in range(len(datasets))]
target_s = [None for i in range(len(datasets))]
target_names = [None for i in range(len(datasets))]
feature_names = [None for i in range(len(datasets))]
description  = [None for i in range(len(datasets))]
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


param_Grid = {"max_features":np.linspace(0.001,0.999, num=10), "max_samples":np.linspace(0.001,0.999, num=10), "n_estimators":np.arange(1, 100)}
#param_Grid = {"n_neighbors":np.arange(1,40, 1), "p":np.arange(1,5), "leaf_size":np.arange(1, 31, 1)}
#param_Grid = {"learning_rate":["optimal", 'invscaling', "adaptive", 'constant'], "l1_ratio":np.linspace(0,1, num = 11), "alpha":np.logspace(-4,-1,num = 40), "loss":["hinge", 'log', 'perceptron', "modified_huber","squared_hinge"]}
#param_Grid = {"C":np.logspace(-3.5, 4, num=100), "gamma":np.logspace(-5, 2.5, num = 100), "kernel":["rbf", "poly", "sigmoid", "linear"]}
#random_Grid = {"C":loguniform(1e-1, 1e3), "gamma":loguniform(1e-5, 1e0), "kernel":["rbf", "poly", "sigmoid", 'linear']}


x_axis = [1250, 1500, 1750, 2000, 2500]

for total in x_axis:
    n_itersearch = total
    results_grid = {}
    results_rand = {}

    rand_test_score = {}
    grid_test_score = {}

    rand_train_score = {}
    grid_train_score = {}

    grid_score = {}
    rand_score = {}

    grid_dict = {}
    rand_dict = {}

    time_rand = {}
    time_grid = {}
    #time_auto = {}
    #auto_stats = {}

    trainrate = 0.75
    num = 10
    tests = {}
    turn = 10

    for i, (name, data) in enumerate(zip(names, datasets)):
        model =  RandomForestClassifier(n_jobs = -1)
        #automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=30)
        # preprocessing 
        if(not(target_names[i] is None)):
            print(f"J'ai commencé le traitement du dataset {names[i]}")
            pre_process = StandardScaler()
            x_train, x_test, y_train, y_test = train_test_split(data_s[i], target_s[i],shuffle=False, train_size= trainrate)
            x_train, x_test = pre_process.fit_transform(x_train), pre_process.fit_transform(x_test)
            #creation des grilles de recherches structurées et aleatoires 
            #grid_t = GridSearchCV(model, param_grid=param_Grid, cv=4, n_jobs=-1, verbose=4)
            rand_t = RandomizedSearchCV(model, param_distributions= param_Grid, n_iter=n_itersearch, cv=3, n_jobs=-1)
            #tests[names[i]] = [x_train, x_test, y_train, y_test]

            train_liste = [0 for _ in range(turn)]
            test_liste = [0 for _ in range(turn)]
            time_liste = [0 for _ in range(turn)]
            best = 0
            for e in range(turn):

                #entrainement et mesure du temps pour le grid_search
                """start = time()
                grid_t.fit(x_train, y_train)
                time_grid[names[i]] = time() - start
                results_grid[names[i]] = grid_t.best_params_"""


                #entrainement et mesure du temps pour le random_search
                start = time()
                rand_t.fit(x_train, y_train)
                time_liste[e] = time() - start
                train_liste[e] = rand_t.best_score_
                #time_rand[names[i]] = time() - start
                if best<rand_t.best_score_:
                    results_rand[names[i]] = rand_t.best_params_
                    best = rand_t.best_score_


                #gridsearch
                #grid_dict[names[i]] = grid_t.cv_results_

                #randomsearch
                #rand_dict[names[i]] = rand_t.cv_results_

                #scoring grid 
                """model.set_params(**grid_t.best_params_)
                model.fit(x_train, y_train)
                results_grid[names[i]]['Test score'] = model.score(x_test, y_test)
                results_grid[names[i]]["Score optimisation"] =  grid_t.best_score_
                results_grid[names[i]]["Temps d'execution(s)"] = time()-start"""

                #scoring rand
                model.set_params(**rand_t.best_params_)
                model.fit(x_train, y_train)
                test_liste[e] = model.score(x_test, y_test) 
            results_rand[names[i]]['test_score'] = np.mean(test_liste)
            results_rand[names[i]]['std_test'] = np.std(test_liste)
            results_rand[names[i]]['train_score'] = np.mean(train_liste)
            results_rand[names[i]]['std_train'] = np.std(train_liste)
            results_rand[names[i]]["time"] = np.mean(time_liste)
            #results_rand[names[i]]["std_time"] = np.std(time_liste)


            print(f"J'ai fini le traitement du dataset {names[i]}")
    pd.DataFrame(results_rand).to_csv(f"RANDSEARCH-RF-{str(total)}")
