import random
from random import *
from time import time

# from matplotlib import  pyplot as plt
import numpy as np
from deap import base
from deap import creator
from deap import tools
from sklearn.datasets import *
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import pickle

kernel = ["linear", "rbf", "poly", "sigmoid"]


def evalOneMax(value):
    loss = ['hinge', 'log', 'perceptron', 'modified_huber', "squared_hinge"]
    learning_rate = ["constant", 'optimal', 'adaptive', 'invscaling']
    model = SGDClassifier(n_jobs=-1, eta0=0.00001, loss=loss[round(abs(value[0] * 6)) % 4],
                          learning_rate=learning_rate[round(abs(value[1] * 5)) % 3], l1_ratio=abs(value[2] % 1),
                          alpha=10 ** (-3 * value[3]))
    scores = cross_val_score(model, x_train, y_train, cv=3, n_jobs=-1)
    return scores.mean(),  # Add a comma even if there is only one return value


def score(value):
    loss = ['hinge', 'log', 'perceptron', 'modified_huber', "squared_hinge"]
    learning_rate = ["constant", 'optimal', 'adaptive', 'invscaling']
    model = SGDClassifier(n_jobs=-1, eta0=0.00001, loss=loss[round(abs(value[0] * 6)) % 4],
                          learning_rate=learning_rate[round(abs(value[1] * 5)) % 3], l1_ratio=abs(value[2] % 1),
                          alpha=10 ** (-3 * value[3]))
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)


def update(ind, mu, std):
    for i, mu_i in enumerate(mu):
        ind[i] = random.gauss(mu_i, std)


creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("update", update)
toolbox.register("evaluate", evalOneMax)


# IND_SIZE = 10


def main(ngen):
    loss = ['hinge', 'log', 'perceptron', 'modified_huber', "squared_hinge"]
    learning_rate = ["constant", 'optimal', 'adaptive', 'invscaling']
    IND_SIZE = 10
    start = time()

    # random.seed(64)

    logbook = tools.Logbook()
    logbook.header = "gen", "fitness", 'loss', 'alpha', 'l1_ratio', "learning_rate"

    # interval = (-3,7)
    func = [random(), random(), random(), gauss(0, 0.5)]
    mu = func
    sigma = 0.5
    alpha = 2.0 ** (1.0 / IND_SIZE)

    best = creator.Individual(mu)
    best.fitness.values = toolbox.evaluate(best)
    worst = creator.Individual(func)
    # print(worst)
    best_score = 0
    save = 0
    NGEN = ngen
    for g in range(NGEN):
        toolbox.update(worst, best, sigma)
        worst.fitness.values = toolbox.evaluate(worst)
        if best.fitness <= worst.fitness:
            sigma = sigma * alpha
            best, worst = worst, best
        else:
            sigma = sigma * alpha ** (-0.25)
    logbook.record(gen=g, fitness=best.fitness.values[0], loss=loss[round(abs(best[0] * 6)) % 5],
                   learning_rate=learning_rate[round(abs(best[1] * 5)) % 4], l1_ratio=abs(best[2] % 1),
                   alpha=10 ** (-3 * best[3]), score=score(best))
    if best_score < best.fitness.values[0]:
        best_score = best.fitness.values[0]
        save = best

    # print("Fin de l'algorithme en "+ str(n_iter)+" tours")
    start = time() - start

    return save, start


if __name__ == "__main__":
    seed(100000)
    np.random.seed(100000)
    datasets = [load_breast_cancer(), load_digits(), load_iris(), load_wine()]  # , load_linnerud
    names = ['load_breast_cancer', 'load_digits', 'load_iris', "load_wine"]  # 'load_linnerud'
    data_s = [None for i in range(len(datasets))]
    target_s = [None for i in range(len(datasets))]
    target_names = [None for i in range(len(datasets))]
    feature_names = [None for i in range(len(datasets))]
    description = [None for i in range(len(datasets))]
    tab = {}
    for i in range(len(names)):
        tab[names[i]] = [[0 for _ in range(10)] for k in range(10)]
    for i, dataset in enumerate(datasets):
        data_s[i] = dataset.data
        target_s[i] = dataset.target
        pocket = list(zip(data_s[i], target_s[i]))
        # print(pocket)
        shuffle(pocket)
        data_s[i] = [x[0] for x in pocket]
        target_s[i] = [x[1] for x in pocket]
        # feature_names[i] = dataset.feature_names
        # description[i] = dataset.DESCR
        # target_names[i] = dataset.target_names
    turn = 10
    x_train, x_test, y_train, y_test = 0, 0, 0, 0
    one_results = {}
    start = [0] * 4
    best_score = [0] * 4
    best2 = [0] * 4
    process = [0 for _ in range(turn)]
    for k in range(10):
        for total in range(10):
            print(f"{total + 1} essais ")
            for i in range(len(datasets)):
                x_train, x_test, y_train, y_test = train_test_split(data_s[i], target_s[i], shuffle=False,
                                                                    train_size=0.75)
                x_train, x_test = StandardScaler().fit_transform(x_train), StandardScaler().fit_transform(x_test)
                best, time1 = main(200)
                train_score = evalOneMax(best)[0]
                if best_score[i] < train_score:
                    best_score[i] = train_score
                    best2[i] = best
                # start[i] = start[i] + time1
                tab[names[i]][total][k] = best_score[i]
                """one_results[names[i]] = {"kernel": kernel[round(best2[i][2] % 3)], "C": 10 ** (-4 * best2[i][0] + 4),
                                         'gamma': 10 ** (-7.5 * abs(best2[i][1]) + 2.5),
                                         "test_score": score(best2[i]),
                                         "train_score": best_score[i],
                                         "Time": start[i]}
            pd.DataFrame(one_results).to_csv(f"ONEFIFTH-SVC-{(total + 1) * 25}")"""
    file_name = "ONE-TAB-SGD"
    outfile = open(file_name, "wb")
    print(tab)
    pickle.dump(tab, outfile)
    outfile.close()
