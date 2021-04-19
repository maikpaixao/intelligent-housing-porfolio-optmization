import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")

listings = pd.read_csv('data/simulated_listings.csv')
target = pd.read_csv('data/target_apartments.csv')

#print(listings.describe())

#listings = listings[listings['sold']==0]
listings = listings[listings['rooms'] < 7]
listings = listings[listings['garages'] < 6]

#Standartization
#scaler = StandardScaler()
#listings = scaler.fit_transform(listings)
#listings = pd.DataFrame(listings)

#Correlation matrix
corrMatrix = listings.corr()

matplotlib.rcParams.update({'font.size': 7})
#sns.heatmap(corrMatrix, annot=True)
sns.scatterplot(x=listings['value'], y=listings['time_on_market'])
#sns.barplot(x=listings['interior_quality'], y=listings['value']) 

plt.show()
