import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

sns.set_theme(style="whitegrid")

listings = pd.read_csv('../data/simulated_listings.csv')

listings = listings[listings['sold']==0]
listings = listings[listings['rooms'] < 7]
listings = listings[listings['garages'] < 9]

corrMatrix = listings.corr()

matplotlib.rcParams.update({'font.size': 7})
#sns.heatmap(corrMatrix, annot=True)
#sns.scatterplot(x=listings['value'], y=listings['time_on_market'])
#sns.barplot(x=listings['interior_quality'], y=listings['time_on_market']) 

plt.show()