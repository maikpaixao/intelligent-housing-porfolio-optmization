import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from matplotlib import pyplot as plt

if __name__ == '__main__':
  listings = pd.read_csv('data/simulated_listings.csv')

  kmf = KaplanMeierFitter()
  kmf.fit(listings['time_on_market'], event_observed=listings['sold'])
  
  kmf.plot()
  plt.show()