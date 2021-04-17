import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing

listings = pd.read_csv('data/simulated_listings.csv')
target = pd.read_csv('data/target_apartments.csv')


listings = listings[listings['sold']==1]
#listings = listings[listings['rooms'] < 7]
#listings = listings[listings['garages'] < 6]

x = listings.iloc[:, :5]

'''
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

x = pd.DataFrame(x_scaled)#.values
print(x.head())
'''
#y = listings['value']
y = listings['time_on_market']

#reg = LinearRegression().fit(x, y)
reg = RandomForestRegressor().fit(x, y)
print(reg.score(x, y))