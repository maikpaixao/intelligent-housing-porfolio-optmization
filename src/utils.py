import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
import pickle

class Utils:
  def __init__(self, filepath):
    self.target = pd.read_csv(filepath)
    sns.set_theme(style="whitegrid")

  def transform_tom(self, listings):
    listings_cpy = listings.copy()
    listings_cpy['time_on_market'] = listings_cpy['time_on_market'].apply(int).apply(range).apply(list)

    tm = listings_cpy.explode(column='time_on_market')#.reset_index()
    #tm['sold'] = tm['time_on_market'].apply(ls)

    tm_positive = tm[tm['sold']==1]
    tm_idx = tm_positive.index
    tm['sold'] = 0

    listing_positive = listings[listings['sold']==1]
    listing_idx = listing_positive.index
    listing_idx = list(listing_idx)

    for idx in listing_idx:
      tm.loc[idx]['sold'].iloc[-1] = 1
    return tm

  def train(self, listings, target):
    model = RandomForestRegressor(random_state=2)
    model.fit(listings, target)

    with open('./models/'+ str(target.name) +'.pkl', 'wb') as file:
      pickle.dump(model, file)
    return model
  
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
    return listings[~((listings < (Q1 - 1.4 * IQR)) | (listings > (Q3 + 1.4 * IQR))).any(axis=1)]
  
  def get_importance(self, model):
    importance = model.feature_importances_
    return importance

  def generate(self, model_p, model_t):
    self.target = self.rearranje_cols(self.target)
    y = self.target['value'].values
    y_hat = model_p.predict(self.target.iloc[:, :6])
    y_tom = model_t.predict(self.target.iloc[:, :7])

    profit = (y_hat - y)/y_tom

    self.target['sell_value'] = y_hat
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
    corrMatrix = data.iloc[:, :-1].corr()
    matplotlib.rcParams.update({'font.size': 7})
    sns.heatmap(corrMatrix, annot=True)
    plt.savefig('saved/heatmap.png', pad_inches=10)
