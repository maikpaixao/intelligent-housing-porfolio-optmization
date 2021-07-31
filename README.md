## Intelligent Housing Porfolio Optmization

<div align="justify">
The process of selecting a portfolio is composed of two stages: the first is to analyze the historical data and build an idea on the behavior of the price and liquidity of the houses in the future, and the second one uses these insights to optimize the portfolio based on maximum profits. Both Machine Learning and Portfolio Strategies are of great interest in real estate market today. This work attempts to combine both subjects by using data science and machine learning to analyse the correlation between house prices and time on market (liquidity). Besisdes, a survival analysis regression model is built to predict the probabality of house being sold at given time.
 </div>

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
<div align="justify">
Below is a heatmap of the independent variables of the listing of houses and it is possible to see that variables such as number of rooms, number of garages and useful area have a strong correlation with the price of properties. In other words, these variables are responsible for the greatest impact on the final price of a house or apartment.
</div>
<img src='https://i.imgur.com/AQjBUMy.png' width=900/>

<div align="justify">
In this work, an analysis of the relationship between price and liquidity was performed. Liquidity is shown by the number of days a home was for sale on the market before being sold. A linear regression model was used to find the equation of the straight line relating price and liquidity. Next, it is possible to see in the first image that there is a problem of heteroskedastic variables. This means that the remains do not follow a Gaussian distribution. To correct this problem the Box Cox transformation is used. The result can be seen in the second image below.

In the end, it can be said that there is a weak correlation between the price of a property and its liquidity. And that makes perfect sense. A house or apartment with a higher sale value generally takes longer to sell when considering a scale of months.
</div>

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
<div align="justify">
In addition, a deeper analysis was performed using the price and liquidity of a property. Some observations in the dataset show properties that, so far, have not yet been sold. Therefore the variable 'Sold' was treated as censored data and a survival analysis was performed using the Time on Market and Sold variables. Data were transformed so that each observation represented only one day that a particular property was for sale. This pattern continues with the dependent variable 'Sold' as 0 until the day the observation constitutes a property sold. In case a house has been sold, the 'Sold' variable is marked as 1, otherwise it remains as 0 until the end of days it has been on the market.

The following image presents a graph using a Kaplan Meier estimator to generate the survival function distribution. Interpreting the plot, it is possible to explain that the longer a property has been on the market for sale, the greater the likelihood of it being sold in contrast to a property that has been on the market for less time.
</div>
  
<img src='https://i.imgur.com/EsN1VB8.png' width=900/>
