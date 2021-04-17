import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

matplotlib.rcParams.update({'font.size': 7})
sns.set_theme(style="whitegrid")

listings = pd.read_csv('data/simulated_listings.csv')
target = pd.read_csv('data/target_apartments.csv')

'''
listings = listings[listings['sold']==1]
x = listings.iloc[:, :4]
y = listings['time_on_market']
'''

#reg = LinearRegression().fit(x, y)
#print(reg.score(x, y), reg.intercept_)

corrMatrix = listings.corr()

#secondary = listings[listings['rooms']==8]
#print(secondary.head())

#sns.heatmap(corrMatrix, annot=True)
sns.scatterplot(x=listings['time_on_market'], y=listings['value'])
#sns.barplot(x=listings['rooms'], y=listings['time_on_market']) 

plt.show()