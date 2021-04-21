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

#change coluns: value => interior_quality
cols = listings.columns.tolist()
cols = cols[:5] + [cols[6]]  + [cols[5]] + cols[7:]
listings = listings[cols]


listings = listings[listings['sold']==1]
listings = listings[listings['rooms'] < 7]
listings = listings[listings['garages'] < 6]

x = listings.iloc[:, :5]

'''
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

x = pd.DataFrame(x_scaled)#.values
print(x.head())
'''
y_price = listings['value']
y_tom = listings['time_on_market']

model_p = RandomForestRegressor().fit(x, y_price)
model_t = RandomForestRegressor().fit(x, y_tom)
#print(reg.score(x, y_price))

pred = utils.generate(model_p, model_t)
print(pred.shape)

#cumulative sum
pred = pred[pred['value'].cumsum() <= 150000000]
print(pred.shape)
print(pred.head())

importance = utils.get_importance(model_p)

#plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()
