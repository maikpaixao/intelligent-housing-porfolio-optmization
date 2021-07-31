from src.utils import Utils
import pickle
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

def load_models():
  with open('./models/value.pkl', 'rb') as file:
    model_p = pickle.load(file)

  with open('./models/time_on_market.pkl', 'rb') as file:
    model_t = pickle.load(file)
  
  return model_p, model_t

if __name__ == '__main__':
  utils = Utils(filepath = 'data/target_apartments.csv')
  model_p, model_t = load_models()

  pred = utils.generate_profits(model_p, model_t)
  pred = pred[pred['value'].cumsum() <= 100000000]

  pred.to_csv('./optimized_portfolio.csv')
  