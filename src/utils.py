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
  def __init__(self, filepath):
    self.target = pd.read_csv(filepath)
    sns.set_theme(style="whitegrid")
  
  def remove_outiliers(self, listings):
    Q1 = listings.quantile(0.25)
    Q3 = listings.quantile(0.75)
    IQR = Q3 - Q1
    return listings[~((listings < (Q1 - 1.5 * IQR)) | (listings > (Q3 + 1.5 * IQR))).any(axis=1)]
  
  def get_importance(self, model):
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
  
  def get_histograms(self, data):
    cols = data.columns
    for idx, col in enumerate(cols):
      sns.histplot(x = data[col], kde=False)
      plt.xlabel(col)
      plt.savefig('saved/histogram_' + col + '.png', dpi=100)
      
  def get_heatmap(self, data):
    corrMatrix = data.corr()
    matplotlib.rcParams.update({'font.size': 7})
    sns.heatmap(corrMatrix, annot=True)
    plt.figure(figsize=(200,200))
    plt.savefig('saved/heatmap.png', dpi=100)
