from src.utils import Utils
import pickle
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

if __name__ == '__main__':
  utils = Utils(filepath = 'data/target_apartments.csv')

  with open('./models/value.pkl', 'rb') as file:
    model_p = pickle.load(file)

  with open('./models/time_on_market.pkl', 'rb') as file:
    model_t = pickle.load(file)

  #cumulative sum
  pred = utils.generate(model_p, model_t)
  pred = pred[pred['value'].cumsum() <= 150000000]

  print(pred.shape)

  pred = pred[pred['interior_quality'] == 1]
  print(pred.shape)
  pred['interior_quality'] = 3

  pd = model_p.predict(pred.iloc[:, :6])
  pred['sell_value_after'] = pd

  pred = pred[pred['sell_value_after'] > pred['sell_value']]
  print(pred.head())
  print(pred.shape)

  #plot feature importance
  #importance = utils.get_importance(model_p)
  #plt.bar([x for x in range(len(importance))], importance)
  #plt.show()
