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

listings = utils.remove_outiliers(listings)

#listings = listings[listings['sold']==1]

#sns.scatterplot(x=listings['interior_quality'], y=listings['interior_quality'])
#sns.barplot(x=listings['interior_quality'], y=listings['value']) 

#utils.get_histograms(data = listings)
#utils.get_heatmap(data = listings)

#sns.histplot(x=listings['time_on_market'], kde=True, fill=False) 
#sns.boxplot(x=listings['interior_quality'], y=listings['value'])

#plt.show()
