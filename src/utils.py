import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

class Utils:
  def __init__(self):
    self.target = pd.read_csv('data/target_apartments.csv')
  
  def get_importance(self, model):
    # get importance
    importance = model.feature_importances_
    return importance

  def generate(self, model_p, model_t):
    y = self.target['value'].values[0]
    y_hat = model_p.predict(self.target.iloc[:, :5])
    y_tom = model_t.predict(self.target.iloc[:, :5])

    profit = (y_hat - y)/y_tom
    self.target['profit'] = profit
    self.profits_data = self.target.sort_values(by=['profit'], ascending=False)

    return self.profits_data
