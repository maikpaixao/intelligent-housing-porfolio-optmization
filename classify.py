import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from src.utils import Utils

utils = Utils(filepath = 'data/target_apartments.csv')

listings = pd.read_csv('data/simulated_listings.csv')
target = pd.read_csv('data/target_apartments.csv')

listings = utils.rearranje_cols(listings, listings=True)
#listings = utils.remove_outiliers(listings)

#listings = listings[listings['rooms'] < 7]
#listings = listings[listings['garages'] < 6]

model_p = utils.train(listings.iloc[:, :6], listings['value'])
print(model_p.score(listings.iloc[:, :6], listings['value']))

listings = listings[listings['sold']==1]
model_t = utils.train(listings.iloc[:, :7], listings['time_on_market'])
print(model_t.score(listings.iloc[:, :7], listings['time_on_market']))

#cumulative sum
pred = utils.generate(model_p, model_t)
pred = pred[pred['value'].cumsum() <= 150000000]

#plot feature importance
importance = utils.get_importance(model_p)
plt.bar([x for x in range(len(importance))], importance)
plt.show()
