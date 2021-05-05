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

utils = Utils(filepath = 'data/target_apartments.csv')
listings = pd.read_csv('data/simulated_listings.csv')
listings = listings.iloc[:50, :]

#listings.describe().to_csv("data_description.csv")

#Use DOM's as time series
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

print(listing_idx)

for idx in listing_idx:
  tm.loc[idx]['sold'].iloc[-1] = 1

#print('Finsihed')
#tm.to_csv('tranformed_data.csv')

#listings = listings[listings['rooms'] < 7]
#listings = listings[listings['garages'] < 6]

'''

listings = listings[listings['sold']==1]
listings = utils.remove_outiliers(listings)

model = LinearRegression()
model.fit(np.array(listings['value']).reshape(-1, 1), listings['time_on_market'])
m = model.coef_
b = model.intercept_


# Plot outputs
plt.scatter(listings['value'], listings['time_on_market'],  color='black')
#plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

#sns.scatterplot(x=listings['value'], y=listings['time_on_market'])
#sns.lineplot(x=listings['time_on_market'], y=listings['value'])
#sns.barplot(x=listings['useful_area'], y=listings['value']) 

#utils.get_histograms(data = listings)
#utils.get_heatmap(data = listings)

sns.scatterplot(x=listings['value'], y=listings['time_on_market']) 
#m, b = np.polyfit(listings['value'], listings['time_on_market'], 1)

plt.plot(listings['value'], m*listings['value'] + b)

#sns.regplot(x=listings['value'], y=listings['time_on_market'], lowess=True)
#sns.boxplot(x=listings['interior_quality'], y=listings['value'])

plt.show()
'''
#SECOND PART

'''
listings = utils.remove_outiliers(listings)
listings = listings[listings['sold']==1]

listings['value'], _ = stats.boxcox(listings['value'])
#listings['time_on_market'], _ = stats.boxcox(listings['time_on_market'])

#listings['value'] = listings['value'].apply(np.log)
#listings['time_on_market'] = listings['time_on_market'].apply(np.log)


model = linear_model.LinearRegression(normalize=True)
model.fit(np.array(listings['value']).reshape(-1, 1), listings['time_on_market'])
m = model.coef_
b = model.intercept_

plt.scatter(listings['value'], listings['time_on_market'],  color='black')
plt.plot(listings['value'], m*listings['value'] + b)


#sns.histplot(data=listings, x="time_on_market")
plt.show()
'''

sns.histplot(data=listings, x="time_on_market")
plt.show()