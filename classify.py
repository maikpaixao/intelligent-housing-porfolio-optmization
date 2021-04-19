import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing

listings = pd.read_csv('data/simulated_listings.csv')
target = pd.read_csv('data/target_apartments.csv')

#change coluns: value => interior_quality
cols = listings.columns.tolist()
cols = cols[:5] + [cols[6]]  + [cols[5]] + cols[7:]
listings = listings[cols]


#listings = listings[listings['sold']==1]
listings = listings[listings['rooms'] < 7]
listings = listings[listings['garages'] < 6]

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
reg = DecisionTreeRegressor().fit(x, y)
print(reg.score(x, y))

pred = reg.predict([[3.0,2.0,77,-23.6246294,-46.7387574]])
print(pred)

# get importance
importance = reg.feature_importances_

# summarize feature importance
for i,v in enumerate(importance):
	print('Variable: %0d, Score: %.5f' % (i,v))

# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()
