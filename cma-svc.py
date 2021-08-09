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
from scoop import futures

def evalOneMax(value):
    while value[1] < -1.5:
        value[1] = value[1]/2   
    while value[0] > 1.9:
        value[0] = value[0]/2
    model = SVC(C = 10**(3*value[0]), gamma=10**(-3*value[1]), kernel=kernel[round(abs(value[2]*4))%3])
    scores = cross_val_score(model, x_train, y_train, cv = 4, n_jobs=-1)
        #print(value)
    return scores.mean(), #Add a comma even if there is only one return value

def score(value):
    model = SVC(C = 10**(3*value[0]), gamma=10**(-3*value[1]), kernel=kernel[round(abs(value[2]*4))%3])
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

def main():
    random.seed(100000)
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
        shuffle(pocket)
        data_s[i] = [x[0] for x in pocket]
        target_s[i] = [x[1] for x in pocket]
        feature_names[i] = dataset.feature_names
        description[i] = dataset.DESCR
        target_names[i] = dataset.target_names


    #svc
    n_iter = 0
    func_seq = [lambda:random.gauss(0,0.5) , lambda:random.gauss(0,0.5), lambda:random.random()]

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
    toolbox.register("mutate", tools.mutGaussian,mu = 0,sigma = 0.5, indpb=0.2)
    toolbox.register("select", tools.selBest)
    toolbox.register("map", futures.map)
    #pool = multiprocessing.Pool()
    #toolbox.register("map", pool.map)

   

#calcul des performances

    for total in [1, 5, 10, 25, 50, 100, 125, 150, 175, 200, 250]:
        ea_results = {}
        cma_results = {}

        for i in range(len(datasets)):
            #n_iter = 0
            x_train, x_test, y_train, y_test = train_test_split(data_s[i], target_s[i], shuffle=False, train_size=0.75)
            x_train, x_test = StandardScaler().fit_transform(x_train), StandardScaler().fit_transform(x_test)
            toolbox.register("evaluate", evalOneMax)
            #pop = toolbox.population(n=10*N)
            #print(pop)
            hof1 = tools.HallOfFame(50)
            hof2 = tools.HallOfFame(50)
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("std", np.std)
            #stats.register("min", np.min)
            #stats.register("max", np.max)

            CXPB, MUTPB, NGEN, turn = 0.3, 0.2, total, 10
            start = time()
            train_liste = [0 for _ in range(turn)]
            test_liste = [0 for _ in range(turn)]
            time_liste = [0 for _ in range(turn)]
            print("------------------- Data : "+names[i]+" ------------------------") 
            best2 = 0
            best_score = 0
            for k in range(turn):
                strategy = cma.Strategy(centroid=[0,0,0], sigma=0.5, lambda_ = 10)
                toolbox.register("generate", strategy.generate, creator.Individual)
                toolbox.register("update", strategy.update)
                #print("--------turn : "+ str(k+1)+"---------")
                start = time()
                pops = algorithms.eaGenerateUpdate(toolbox, ngen=NGEN, stats=stats, halloffame=hof2, verbose = False)
                #print(len(pops[0]))
                time_liste[k] = time()-start
                pops = pops[0]
                best = pops[np.argmax([toolbox.evaluate(x) for x in pops])]
                score_tmp = evalOneMax(best)[0]
                if best_score < score_tmp:
                    best2 = best
                    best_score = score_tmp
                train_liste[k] = best_score
                test_liste[k] = score(best2) 
            cma_results[names[i]] = {"kernel":kernel[round(best2[2]%3)], "C":10**(3*best2[0]), 'gamma':10**(-3*best2[1]), 'test_score': np.mean(test_liste),'std_test': np.std(test_liste),
                                     "train_score": np.mean(train_liste), "std_train":np.std(train_liste),"Time":np.mean(time_liste)}
        pd.DataFrame(cma_results).to_csv(f"CMA-SVC-{str(total*10)}")

if __name__ == "__main__":
    main()
