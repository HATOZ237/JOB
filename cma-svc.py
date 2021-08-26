import multiprocessing
import pickle
import random
from random import *
from time import time

import numpy as np
import pandas as pd
from deap import algorithms
from deap import base
from deap import cma
from deap import creator
from deap import tools
from sklearn.datasets import *
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

seed(100000)
np.random.seed(100000)
datasets = [load_breast_cancer(), load_digits(), load_iris(), load_wine()]  # , load_linnerud
names = ['load_breast_cancer', 'load_digits', 'load_iris', "load_wine"]  # 'load_linnerud'
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

n_iter = 0
tab = {}
for i in range(len(names)):
    tab[names[i]] = np.array([[0 for _ in range(10)] for k in range(10)])
func_seq = [lambda: random(), lambda: random(), lambda: random()]

x_train, x_test, y_train, y_test = [0] * 4

kernel = ["linear", "rbf", "poly", "sigmoid"]

creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Add a comma even if there is only one argument
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
# Attribute generator
# Structure initializers
toolbox.register("individual", tools.initCycle, creator.Individual,
                 func_seq, n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.3, indpb=0.02)
toolbox.register("select", tools.selBest)

f = lambda x: x[0]


def evalOneMax(value):
    # print(value)
    # lock = value[1]
    model = SVC(C=10 ** (-4 * abs(value[0]) + 4), gamma=10 ** (-7.5 * abs(value[1]) + 2.5),
                kernel=kernel[round(abs(value[2] * 4)) % 3])
    scores = cross_val_score(model, x_train, y_train, cv=3, n_jobs=1)
    # print(value)
    return scores.mean(),  # Add a comma even if there is only one return value


def score(value):
    model = SVC(C=10 ** (-4 * abs(value[0]) + 4), gamma=10 ** (-7.5 * abs(value[1]) + 2.5),
                kernel=kernel[round(abs(value[2] * 4)) % 3])
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)


# calcul des performances
def main(id):
    cma_results = {}
    best_score = [0] * 4
    times = [0] * 4
    for k in range(10):
        for i in range(len(datasets)):
            global x_train, x_test, y_train, y_test
            x_train, x_test, y_train, y_test = train_test_split(data_s[i], target_s[i], shuffle=False, train_size=0.75)
            x_train, x_test = StandardScaler().fit_transform(x_train), StandardScaler().fit_transform(x_test)
            toolbox.register("evaluate", evalOneMax)
            pool = multiprocessing.Pool()
            toolbox.register("map", pool.map)
            # pop = toolbox.population(n=10*N)
            # print(pop)
            # hof1 = tools.HallOfFame(50)
            hof2 = tools.HallOfFame(2)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            stats.register("min", np.min)
            stats.register("max", np.max)

            CXPB, MUTPB, NGEN, turn = 0.3, 0.2, 10, 4
            train_liste = [0 for _ in range(turn)]
            test_liste = [0 for _ in range(turn)]
            time_liste = [0 for _ in range(turn)]

            best2 = 0
            strategy = cma.Strategy(centroid=[random(), random(), random()], sigma=0.3, lambda_=4)
            toolbox.register("generate", strategy.generate, creator.Individual)
            toolbox.register("update", strategy.update)

            # print("--------turn : "+ str(k+1)+"---------")
            start = time()
            pops = algorithms.eaGenerateUpdate(toolbox, ngen=NGEN, stats=stats, halloffame=hof2, verbose=False)
            times[i] = times[i] + time() - start
            best2 = hof2[0]
            pops = hof2
            scores = toolbox.map(toolbox.evaluate, hof2)
            train_liste = list(map(f, scores))
            if best_score[i] < max(train_liste):
                best_score[i] = max(train_liste)
            cma_results[names[i]] = {"kernel": kernel[round(abs(best2[2] * 4)) % 3], "C": 10 ** (-4 * best2[0] + 4),
                                     'gamma': 10 ** (-7.5 * abs(best2[1]) + 2.5),
                                     "max_train_score": best_score[i], 'test_score': score(best2),
                                     "train_score": np.mean(train_liste), "std_train": np.std(train_liste),
                                     "Time": times[i]}
            global tab
            print(best_score[i])
            tab[names[i]][k, id] = best_score[i]
        pd.DataFrame(cma_results).to_csv(f"CMAS-SVC-{str((k + 1) * 20)}")


if __name__ == "__main__":
    for id in range(1):
        print("------------------- Tour  : " + str(id) + " ------------------------")
        main(id)
    file_name = "CMA-TAB-SVC"
    outfile = open(file_name, "wb")
    pickle.dump(tab, outfile)
    outfile.close()
