## Intelligent Housing Porfolio Optmization

the process of selecting a portfolio is composed of two stages: the first is to analyze the historical data and build an idea on the behavior of the price and liquidity of the houses in the future, and the second one uses these insights to optimize the portfolio based on maximum profits. Both Machine Learning and Portfolio Strategies are of great interest in real estate market today. This work attempts to combine both subjects by using data science and machine learning to analyse the correlation between house prices and time on market (liquidity). Besisdes, a survival analysis regression model is built to predict the probabality of house being sold at given time.

### Dependencies
```bash
scikit-learn==0.24.2
scipy==1.1.0
pandas==1.1.5
numpy==1.19.5
lifelines==0.26.0
matplotlib==3.2.1
```

### Linear Regression Analysis
```bash
python regression.py
```
<img src='https://i.imgur.com/AQjBUMy.png' width=900/>

<p float="left">
<img src='https://i.imgur.com/DJnaCCV.png' width=400/>
<img src='https://user-images.githubusercontent.com/12498735/127746861-34de68f0-d72a-4bff-bffe-6415a499b024.png' width=400/>
</p>

### Training
```bash
python train.py
```

### Survival Regression Analysis
```bash
python survival.py
```
<img src='https://i.imgur.com/EsN1VB8.png' width=900/>
