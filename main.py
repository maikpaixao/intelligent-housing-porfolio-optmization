import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

sns.set_theme(style="whitegrid")

listings = pd.read_csv('data/simulated_listings.csv')
target = pd.read_csv('data/target_apartments.csv')

listings = listings[listings['sold']==1]
#listings = listings[listings['rooms'] < 7]
#listings = listings[listings['garages'] < 6]

corrMatrix = listings.corr()

matplotlib.rcParams.update({'font.size': 6})
sns.heatmap(corrMatrix, annot=True)
#sns.scatterplot(x=listings['time_on_market'], y=listings['value'])
#sns.barplot(x=listings['rooms'], y=listings['time_on_market']) 

plt.show()