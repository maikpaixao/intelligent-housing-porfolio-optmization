import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from src.utils import Utils
import pickle

utils = Utils(filepath = 'data/target_apartments.csv')

listings = pd.read_csv('data/simulated_listings.csv')
target = pd.read_csv('data/target_apartments.csv')

listings = utils.rearranje_cols(listings, listings=True)
listings = utils.remove_outiliers(listings)

listings = listings[listings['sold']==1]
model_p = utils.train(listings.iloc[:, :6], listings['value'])
model_t = utils.train(listings.iloc[:, :7], listings['time_on_market'])
