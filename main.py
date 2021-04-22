import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler
from src.utils import Utils

utils = Utils(filepath = 'data/target_apartments.csv')
listings = pd.read_csv('data/simulated_listings.csv')
#listings.describe().to_csv("data_description.csv")

#listings = listings[listings['rooms'] < 7]
#listings = listings[listings['garages'] < 6]

listings = listings[listings['sold']==1]
listings = utils.remove_outiliers(listings)

model = LinearRegression()
model.fit(np.array(listings['value']).reshape(-1, 1), listings['time_on_market'])
m = model.coef_
b = model.intercept_

'''
# Plot outputs
plt.scatter(listings['value'], listings['time_on_market'],  color='black')
#plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()
'''

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
