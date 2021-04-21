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
sns.set_theme(style="whitegrid")

listings = pd.read_csv('data/simulated_listings.csv')
#listings.describe().to_csv("data_description.csv")
print(listings.shape)

'''
listings = utils.remove_outiliers(listings)

listings = listings[listings['sold']==1]

#Correlation matrix
corrMatrix = listings.corr()

matplotlib.rcParams.update({'font.size': 7})
#sns.heatmap(corrMatrix, annot=True)
#sns.scatterplot(x=listings['interior_quality'], y=listings['interior_quality'])
#sns.barplot(x=listings['interior_quality'], y=listings['value']) 

utils.get_histograms(data = listings)

#sns.histplot(x=listings['time_on_market'], kde=True, fill=False) 
#sns.boxplot(x=listings['interior_quality'], y=listings['value'])
#plt.show()
'''
