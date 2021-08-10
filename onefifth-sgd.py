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

def evalOneMax(value):
    #print(value)
    model = SGDClassifier(n_jobs=-1,eta0=0.0001, loss=loss[round(abs(value[0]*6))%5], learning_rate=learning_rate[round(abs(value[1]*5))%4], l1_ratio=abs(value[2]%1), alpha=10**(-3*value[3]))
    scores = cross_val_score(model, x_train, y_train, cv = 3, n_jobs=-1)
    return scores.mean(), #Add a comma even if there is only one return value

def score(value):
    model = SGDClassifier(n_jobs=-1,eta0=0.0001, loss=loss[round(abs(value[0]*6))%5], learning_rate=learning_rate[round(abs(value[1]*5))%4], l1_ratio=abs(value[2]%1), alpha=10**(-3*value[3]))
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

IND_SIZE = 30
n_iter = 0
func = [random.random(), random.random(), random.random(), random.gauss(0, 0.5)]
loss = ['hinge', 'log', 'perceptron', 'modified_huber', "squared_hinge"]
learning_rate = ["constant", 'optimal', 'adaptive', 'invscaling']
def main():
    start = time()
    random.seed(64)
    
    logbook = tools.Logbook()
    logbook.header = "gen", "fitness", 'loss', 'alpha', 'l1_ratio',"learning_rate", "score"

    #interval = (-3,7)
    mu = func
    sigma = 0.5
    alpha = 2.0**(1.0/IND_SIZE)

    best = creator.Individual(mu)
    best.fitness.values = toolbox.evaluate(best)
    worst = creator.Individual(func)
    #print(worst)

    NGEN = 10
    for g in range(NGEN):
        toolbox.update(worst, best, sigma) 
        worst.fitness.values = toolbox.evaluate(worst)
        if best.fitness <= worst.fitness:
            sigma = sigma * alpha
            best, worst = worst, best
        else:
            sigma = sigma * alpha**(-0.25)
            
        logbook.record(gen=g, fitness=best.fitness.values[0], loss=loss[round(abs(best[0]*6))%5], learning_rate=learning_rate[round(abs(best[1]*5))%4], l1_ratio=abs(best[2]%1), alpha=10**(-3*best[3]), score=score(best))
        print(logbook.stream)
    print("Fin de l'algorithme en "+ str(n_iter)+" tours")
    start = time()-start
    return best, start
    
if __name__ == "__main__":
    one_results = {}
    for i in range(len(datasets)):
        n_iter = 0
        x_train, x_test, y_train, y_test = train_test_split(data_s[i], target_s[i], shuffle=False, train_size=0.75)
        x_train, x_test = StandardScaler().fit_transform(x_train), StandardScaler().fit_transform(x_test)
        best2, time1 = main()
        one_results[names[i]] = {'loss':loss[round(abs(best2[0]*6))%5], "learning_rate":learning_rate[round(best2[1]%3)], 'l1_ratio':abs(best2[2]%1),"alpha":10**(-3*best2[3]), "train_score": evalOneMax(best2)[0],'test_score': score(best2), "Temps d'exec(s)":time1, "Nbre d'évaluations":n_iter}
