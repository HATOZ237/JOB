import multiprocessing
import pickle
from multiprocessing import Process
from random import *
from time import time

# from matplotlib import  pyplot as plt
import numpy as np
from deap import base
from deap import creator
from deap import tools
from sklearn.datasets import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import multiprocessing

kernel = ["linear", "rbf", "poly", "sigmoid"]


def evalOneMax(value):
    if abs(value[2]) > 1:
        value[2] = random()
    model = RandomForestClassifier(n_estimators=round(abs(value[2]) * 100) + 1, max_features=abs(value[0]) % 1,
                                   max_samples=abs(value[1]) % 1, n_jobs=1)
    scores = cross_val_score(model, x_train, y_train, cv=3, n_jobs=1)
    return scores.mean(),  # Add a comma even if there is only one return value


def score(value):
    if abs(value[2]) > 1:
        value[2] = random()
    model = RandomForestClassifier(n_estimators=round(abs(value[2]) * 100) + 1, max_features=abs(value[0]) % 1,
                                   max_samples=abs(value[1]) % 1, n_jobs=1)
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)


def update(ind, mu, std):
    for i, mu_i in enumerate(mu):
        ind[i] = gauss(mu_i, std)

def sorted(liste):
    best_value = liste[0]
    for i in range(len(liste)):
        if best_value < liste[i]:
            best_value = liste[i]
        else :
            liste[i] = best_value
    return liste

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("update", update)
toolbox.register("evaluate", evalOneMax)


# IND_SIZE = 10


def main(id):
    IND_SIZE = 10
    start = time()

    # random.seed(64)

    logbook = tools.Logbook()
    logbook.header = "gen", "fitness", "id"

    # interval = (-3,7)
    func = [gauss(0, 0.5), gauss(0, 0.5), gauss(0, 0.4)]
    mu = func
    sigma = 0.5
    alpha = 2.0 ** (1.0 / IND_SIZE)

    best = creator.Individual(mu)
    best.fitness.values = toolbox.evaluate(best)
    worst = creator.Individual(func)
    # print(worst)
    best_score = 0
    save = 0
    NGEN = 200
    for g in range(NGEN):
        toolbox.update(worst, best, sigma)
        worst.fitness.values = toolbox.evaluate(worst)
        if best.fitness <= worst.fitness:
            sigma = sigma * alpha
            best, worst = worst, best
        else:
            sigma = sigma * alpha ** (-0.25)
    logbook.record(gen=g, fitness=best.fitness.values[0], id = id)
    print(logbook.stream)
    # print("Fin de l'algorithme en "+ str(n_iter)+" tours")
    if best_score < best.fitness.values[0]:
        best_score = best.fitness.values[0]
        save = best

    # print("Fin de l'algorithme en "+ str(n_iter)+" tours")
    start = time() - start

    scores[id] = best_score


if __name__ == "__main__":
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
        data_s[i] = StandardScaler().fit_transform(dataset.data)
        target_s[i] = dataset.target
        pocket = list(zip(data_s[i], target_s[i]))
        # print(pocket)
        np.random.shuffle(pocket)
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
        scores = multiprocessing.Array('d', 10)
        np.random.seed(randint(1, 100000))
        print(f"{k + 1} essais ")
        for i in range(len(datasets)):
            x_train, x_test, y_train, y_test = train_test_split(data_s[i], target_s[i], shuffle=False,
                                                                train_size=0.75, random_state=0)
            # x_train, x_test = StandardScaler().fit_transform(x_train), StandardScaler().fit_transform(x_test)
            for x in range(turn):
                process[x] = Process(target = main, args = (x,))
            start = time()
            for x in range(turn):
                process[x].start()
            for x in range(turn):
                process[x].join()
            #best, time1 = main(200)

            """if best_score[i] < best:
                best_score[i] = best
                best2[i] = best"""
            # start[i] = start[i] + time1
            tab[names[i]][k] = list(sorted(scores))
            """one_results[names[i]] = {"kernel": kernel[round(best2[i][2] % 3)], "C": 10 ** (-4 * best2[i][0] + 4),
                                     'gamma': 10 ** (-7.5 * abs(best2[i][1]) + 2.5),
                                     "test_score": score(best2[i]),
                                     "train_score": best_score[i],
                                     "Time": start[i]}
        pd.DataFrame(one_results).to_csv(f"ONEFIFTH-SVC-{(total + 1) * 25}")"""
file_name = "ONE-TAB-RF"
outfile = open(file_name, "wb")
print(tab)
pickle.dump(tab, outfile)
outfile.close()
