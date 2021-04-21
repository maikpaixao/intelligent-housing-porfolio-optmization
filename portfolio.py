from src.utils import Utils
import pickle
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

utils = Utils(filepath = 'data/target_apartments.csv')

with open('./models/value.pkl', 'rb') as file:
  model_p = pickle.load(file)

with open('./models/time_on_market.pkl', 'rb') as file:
  model_t = pickle.load(file)

#cumulative sum
pred = utils.generate(model_p, model_t)
pred = pred[pred['value'].cumsum() <= 150000000]

print(pred.shape)

#plot feature importance
importance = utils.get_importance(model_p)
plt.bar([x for x in range(len(importance))], importance)
plt.show()
