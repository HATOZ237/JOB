import sklearn
from random import *
#from matplotlib import  pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from numpy.linalg import *
from sklearn.datasets import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, validation_curve, cross_val_score
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
import multiprocessing
from multiprocessing import Process



    




kernel = ["linear", "rbf", "poly","sigmoid"]

def evalOneMax(value):
    #lock = value[1]
    #print(value)
    while value[1] < -1.5:
        value[1] = value[1]/2   
    while value[0] > 1.9:
        value[0] = value[0]/2
    model = SVC(C = 10**(3*value[0]), gamma=10**(-3*value[1]), kernel=kernel[round(abs(value[2]*4))%3])
    scores = cross_val_score(model, x_train, y_train, cv = 3, n_jobs=1)
    #print(value)
    return scores.mean(), #Add a comma even if there is only one return value

def score(value):
    model = SVC(C = 10**(3*value[0]), gamma=10**(-3*value[1]), kernel=kernel[round(abs(value[2]*4))%3])
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

def update(ind, mu, std):
    for i, mu_i in enumerate(mu):
        ind[i] = random.gauss(mu_i,std)
 

creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()                  
toolbox.register("update", update)
toolbox.register("evaluate", evalOneMax)

#IND_SIZE = 10




def main(ngen, id):
    IND_SIZE = 10
    
    #random.seed(64)
    
    #logbook = tools.Logbook()
    #logbook.header = "gen", "fitness", 'loss', 'alpha', 'l1_ratio',"learning_rate", "score"

    #interval = (-3,7)
    func = [random.gauss(0,0.5) , random.gauss(0,0.5), random.random()]
    mu = func
    sigma = 0.5
    alpha = 2.0**(1.0/IND_SIZE)

    best = creator.Individual(mu)
    best.fitness.values = toolbox.evaluate(best)
    worst = creator.Individual(func)
    #print(worst)

    NGEN = ngen
    for g in range(NGEN):
        toolbox.update(worst, best, sigma) 
        worst.fitness.values = toolbox.evaluate(worst)
        if best.fitness <= worst.fitness:
            sigma = sigma * alpha
            best, worst = worst, best
        else:
            sigma = sigma * alpha**(-0.25) 
        #logbook.record(gen=g, fitness=best.fitness.values[0], loss=loss[round(abs(best[0]*6))%5], learning_rate=learning_rate[round(abs(best[1]*5))%4], l1_ratio=abs(best[2]%1), alpha=10**(-3*best[3]), score=score(best))
        #print(logbook.stream)
    #print("Fin de l'algorithme en "+ str(n_iter)+" tours")
    #start = time()-start
    global train_liste, test_liste, time_liste
    train_liste[id] = evalOneMax(best)[0]
    test_liste[id] = score(best)
    #time_liste[id] = time() - start
    #return best, start
    
if __name__ == "__main__":
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
    turn  = 10
    x_train, x_test, y_train, y_test = 0,0,0,0
    one_results = {}
    train_liste = multiprocessing.Array('d', turn)
    test_liste = multiprocessing.Array('d', turn)
    start = 0
    process = [0 for _ in range(turn)]
    for total in [10, 50, 100, 250, 500, 750, 1000,  1250, 1500, 1750, 2000, 2500]:
        print(f"{total} essais demarré")
        for i in range(len(datasets)):
            x_train, x_test, y_train, y_test = train_test_split(data_s[i], target_s[i], shuffle=False, train_size=0.75)
            x_train, x_test = StandardScaler().fit_transform(x_train), StandardScaler().fit_transform(x_test)
            for x in range(turn):
                process[x] = Process(target = main, args = (total,x))
            start = time()
            for x in range(turn):
                process[x].start()
            for x in range(turn):
                process[x].join()
            start = time() - start
        
            one_results[names[i]] = {"max_test_score": np.max(test_liste), "max_train_score": np.max(train_liste), "test_score": np.mean(test_liste), "std_test":np.std(test_liste),"train_score":np.mean(train_liste) ,'std_train':np.std(test_liste) , "Time":start/turn}
        pd.DataFrame(one_results).to_csv(f"ONEFIFTHS-SVC-{total}")
        print(f"{total} essais terminé")
