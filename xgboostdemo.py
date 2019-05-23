import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import operator
import matplotlib.pyplot as plt
import seaborn as sns
import lime.lime_tabular
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
import numpy as np
from sklearn.grid_search import GridSearchCV



data = pd.read_csv("./data/titantic/train.csv")
y = data.Survived
X = data.drop(["Survived", "Name", "PassengerId"], 1)
X = pd.get_dummies(X)