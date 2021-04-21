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
listings = utils.remove_outiliers(listings)


listings = listings[listings['sold']==1]
listings = listings[listings['rooms'] < 7]
listings = listings[listings['garages'] < 6]

x = listings.iloc[:, :6]

y_price = listings['value']
y_tom = listings['time_on_market']

model_p = RandomForestRegressor().fit(x, y_price)
model_t = RandomForestRegressor().fit(x, y_tom)

print(model_p.score(x, y_price), model_t.score(x, y_tom))

pred = utils.generate(model_p, model_t)

#cumulative sum
pred = pred[pred['value'].cumsum() <= 150000000]
print(pred.shape)
print(pred.head())

importance = utils.get_importance(model_p)

#plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()
