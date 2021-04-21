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
  
  def rearranje_cols(self, data, listings=False):
    if listings:
      cols = data.columns.tolist()
      cols = cols[:5] + [cols[6]]  + [cols[5]] + cols[7:]
    else:
      cols = data.columns.tolist()
      cols = cols[:5] + [cols[6]]  + [cols[5]]
    return data[cols]
  
  def remove_outiliers(self, listings):
    Q1 = listings.quantile(0.25)
    Q3 = listings.quantile(0.75)
    IQR = Q3 - Q1
    return listings[~((listings < (Q1 - 1.5 * IQR)) | (listings > (Q3 + 1.5 * IQR))).any(axis=1)]
  
  def get_importance(self, model):
    importance = model.feature_importances_
    return importance

  def generate(self, model_p, model_t):
    self.target = self.rearranje_cols(self.target)
    print(self.target.head())
    y = self.target['value'].values
    y_hat = model_p.predict(self.target.iloc[:, :6])
    y_tom = model_t.predict(self.target.iloc[:, :6])

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
    plt.savefig('saved/heatmap.png', dpi=100)
