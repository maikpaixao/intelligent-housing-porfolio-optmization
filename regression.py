import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from src.utils import Utils
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
  utils = Utils(filepath = 'data/target_apartments.csv')
  listings = pd.read_csv('data/simulated_listings.csv')

  listings = utils.remove_outiliers(listings)
  listings = listings[listings['sold']==1]

  listings['value'], _ = stats.boxcox(listings['value'])
  listings['time_on_market'], _ = stats.boxcox(listings['time_on_market'])

  model = linear_model.LinearRegression(normalize=True)
  model.fit(np.array(listings['value']).reshape(-1, 1), listings['time_on_market'])
  m = model.coef_
  b = model.intercept_

  plt.scatter(listings['value'], listings['time_on_market'],  color='black')
  plt.plot(listings['value'], m*listings['value'] + b)
  plt.show()