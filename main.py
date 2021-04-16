import pandas as pd
import numpy as np

listings = pd.read_csv('data/simulated_listings.csv')
target = pd.read_csv('data/target_apartments.csv')

print(listings.isnull().values.any())

