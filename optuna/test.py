import os
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import optuna
from optuna.integration.mlflow import MLflowCallback
 
iris = load_iris()
trainx, testx, trainy, testy = train_test_split(iris.data, iris.target, test_size=0.2)
 
def objective(trial):
    gamma = trial.suggest_loguniform('gamma', 1e-3, 3.0)
    C = trial.suggest_loguniform('C', 1e+0, 1e+2/2)
    kernel = trial.suggest_categorical('kernel', ['linear','rbf','sigmoid'])
    svc = SVC(gamma=gamma, C=C, kernel=kernel)
    svc.fit(trainx, trainy)
    predy = svc.predict(testx)
    accuracy = accuracy_score(testy, predy)
    return accuracy
 
if __name__=='__main__':
 
    mlflc = MLflowCallback(tracking_uri='ml_exp',
                      metric_name='accuracy')
    study = optuna.create_study(study_name='iris_test')
    study.optimize(objective, n_trials=50, callbacks=[mlflc])
