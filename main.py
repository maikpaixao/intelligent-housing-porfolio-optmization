import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

listings = pd.read_csv('data/simulated_listings.csv')
target = pd.read_csv('data/target_apartments.csv')

corrMatrix = listings.corr()

matplotlib.rcParams.update({'font.size': 7})

#sns.heatmap(corrMatrix, annot=True)
#sns.scatterplot(x=listings['useful_area'], y=listings['value'])
sns.displot(x=listings['rooms'])

plt.show()