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

#change coluns: value => interior_quality
cols = listings.columns.tolist()
cols = cols[:5] + [cols[6]]  + [cols[5]] + cols[7:]
listings = listings[cols]

'''
listings = listings[listings['sold']==0]
#listings = listings[listings['rooms'] < 7]
#listings = listings[listings['garages'] < 6]

corrMatrix = listings.corr()

matplotlib.rcParams.update({'font.size': 7})
#sns.heatmap(corrMatrix, annot=True)
#sns.scatterplot(x=listings['value'], y=listings['interior_quality'])
sns.barplot(x=listings['interior_quality'], y=listings['value']) 

plt.show()
'''