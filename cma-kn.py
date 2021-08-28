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
import pickle

np.random.seed(100000)
datasets = [load_breast_cancer(), load_digits(), load_iris(), load_wine()]#, load_linnerud
names = ['load_breast_cancer', 'load_digits', 'load_iris', "load_wine"]# 'load_linnerud'
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
    np.random.shuffle(pocket)
    data_s[i] = [x[0] for x in pocket]
    target_s[i] = [x[1] for x in pocket]
    feature_names[i] = dataset.feature_names
    description[i] = dataset.DESCR
    target_names[i] = dataset.target_names

n_iter = 0
func_seq = [lambda:random.gauss(0,0.5), lambda:random.random(), lambda:random.gauss(0, 0.5)]

x_train, x_test, y_train, y_test = [0]*4

kernel = ["linear", "rbf", "poly","sigmoid"]

creator.create("FitnessMax", base.Fitness, weights=(1.0,)) #Add a comma even if there is only one argument
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
# Attribute generator
# Structure initializers
toolbox.register("individual", tools.initCycle, creator.Individual, 
    func_seq, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)                       

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian,mu = 0,sigma = 0.5, indpb=0.02)
toolbox.register("select", tools.selBest)


def evalOneMax(value):
    #print("ICI")
    if value[0]>1:
        value[0] = random()
    if value[2]>1:
        value[2] = random()
    model = KNeighborsClassifier(n_neighbors=round(abs(value[0])*40)+1, p=round(abs(value[1])*5)+1, leaf_size=round(abs(value[2]*30))+1, n_jobs=-1)
    scores = cross_val_score(model, x_train, y_train, cv = 3, n_jobs=--1)
    return scores.mean(), #Add a comma even if there is only one return value

def score(value):
    if value[0]>1:
        value[0] = random()
    if value[2]>1:
        value[2] = random()
    model = KNeighborsClassifier(n_neighbors=round(abs(value[0])*40)+1, p=round(abs(value[1])*5)+1, leaf_size=round(abs(value[2])*30)+1, n_jobs=-1)
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

tab = {}
for i in range(len(names)):
    tab[names[i]] = [[0 for _ in range(10)] for k in range(10)]

f = lambda x: x[0]
# calcul des performances
def main(idi):
    cma_results = {}
    best_score = [0] * 4
    times = [0] * 4
    for k in range(10):
        for i in range(len(datasets)):
            global x_train, x_test, y_train, y_test
            x_train, x_test, y_train, y_test = train_test_split(data_s[i], target_s[i], shuffle=False, train_size=0.75, random_state=0)
            x_train, x_test = StandardScaler().fit_transform(x_train), StandardScaler().fit_transform(x_test)
            toolbox.register("evaluate", evalOneMax)
            #pool = multiprocessing.Pool()
            #toolbox.register("map", pool.map)
            # pop = toolbox.population(n=10*N)
            # print(pop)
            # hof1 = tools.HallOfFame(50)
            hof2 = tools.HallOfFame(50)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)

            CXPB, MUTPB, NGEN, turn = 0.3, 0.2, 50, 4
            train_liste = [0 for _ in range(turn)]
            test_liste = [0 for _ in range(turn)]
            time_liste = [0 for _ in range(turn)]

            best2 = 0
            strategy = cma.Strategy(centroid=[random(), random(), random()], sigma=0.3, lambda_=4)
            toolbox.register("generate", strategy.generate, creator.Individual)
            toolbox.register("update", strategy.update)

            # print("--------turn : "+ str(k+1)+"---------")
            start = time()
            pops = algorithms.eaGenerateUpdate(toolbox, ngen=NGEN, stats=stats, halloffame=hof2, verbose=True)
            times[i] = times[i] + time() - start
            best2 = hof2[0]
            pops = hof2
            scores = toolbox.map(toolbox.evaluate, hof2)
            train_liste = list(map(f, scores))
            print(train_liste)
            if best_score[i] < max(train_liste):
                best_score[i] = max(train_liste)
            cma_results[names[i]] = {
                                     "max_train_score": best_score[i], 'test_score': score(best2),
                                     "train_score": np.mean(train_liste), "std_train": np.std(train_liste),
                                     "Time": times[i]}
            global tab
            print(best_score[i])
            tab[names[i]][k][idi] = best_score[i]
        pd.DataFrame(cma_results).to_csv(f"CMAS-SGD-{str((k + 1) * 20)}")


if __name__ == "__main__":
    for id in range(10):
        print("------------------- Tour  : " + str(id) + " ------------------------")
        np.random.seed(randint(1, 100000))
        main(id)
    file_name = "CMA-TAB-KN"
    outfile = open(file_name, "wb")
    print(tab)
    pickle.dump(tab, outfile)
    outfile.close()
