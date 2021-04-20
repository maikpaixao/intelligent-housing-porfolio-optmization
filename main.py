import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from scipy.stats import zscore
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid")

listings = pd.read_csv('data/simulated_listings.csv')
target = pd.read_csv('data/target_apartments.csv')

print(listings.shape)

#print(listings.describe())

#listings = listings[listings['sold']==1]
#listings = listings[listings['rooms'] < 7]
#listings = listings[listings['garages'] < 6]

#Standartization
#scaler = StandardScaler()
#listings = scaler.fit_transform(listings)
#listings = pd.DataFrame(listings)


#remove outliers
z_scores = zscore(listings)
abs_z_scores = np.abs(z_scores)
listings = listings[(abs_z_scores < 3).all(axis=1)]
print(listings.shape)

#listings = listings[listings['sold']==1]

#Correlation matrix
corrMatrix = listings.corr()

matplotlib.rcParams.update({'font.size': 7})
sns.heatmap(corrMatrix, annot=True)
#sns.scatterplot(x=listings['interior_quality'], y=listings['interior_quality'])
#sns.barplot(x=listings['interior_quality'], y=listings['time_on_market']) 
print(listings.head())
#sns.boxplot(x=listings['interior_quality'], y=listings['value'])

plt.show()
