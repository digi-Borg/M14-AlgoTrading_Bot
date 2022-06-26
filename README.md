# Mod14-AlgoTrading_Bot 

![M14TitleIllus](Images/M14ReadMe_2022-06-26010033.png)

*"Using a MachineLearning Algorithmic Trading Bot that adapts to new data and evolving markets."* 

## Background

This project creates an algorithmic trading bot that learns and adapts to new data and evolving markets for a financial advisory firm. Seeking to maintain an advantage over the competition, initiatives are taken to make improvements to the existing algorithmic trading system. The plan is to enhance the current trading signals by testing different classified machine learning (ML) algorithms in a Python notebook. By testing different classified MLs, performance can be compared to adaptation of new data under differing market environments. 

The FinTech technology in this program utilizes ScikitLearn ML algorithm software tools. The tools are used to build algorithmic models by adjusting inputs for buy, sell or hold signals; these signals are derived from a technical indicator called simple moving average(SMA). SMA's calculate the average stock prices over a rolling time period designating a number of days in time windows. 

Using the pandas rolling function with the `window` parameter the time can be specified and adjusted from days to minutes depending on your trading strategies. A short-window(usually 10, 20, 50 days) and long-window(usually 100,200 days) are designated to illustrate positive or negative price trends. When looking at a chart for example, the short window 20 day SMA crosses over the long-window 100 day SMA a trigger point to enter or exit a trade. These SMA crossover points known as dual-moving-average-crossover(DMAC) signal buys or sells by our algorithm model.  

Adjusting the ML algorithm inputs one can put together models, tune them, visualize performance on charts, and backtest them for evaluation on a Classification Report. Additionaly, other classes of ML models are available for backtesting and predicting outcomes before applying them to your strategies.  


---
## Evaluation Results

The following evaluation describes the performance test of the imported models, with the loss and accuracy metric scores of all algorithm machine learning models. To analyze and conclude each model’s performance it's Cumulative Returns chart is visualized and Classifier Report are evaluated. 

* SVC Original Model: 
  * Original nn features: 2 hiddenlayers 'relu', with 50 epochs
  * Original nn Model Accuracy: 0.7289
  * Original nn Model Loss: 0.5561     
  
* LogisticRegression Model Results:
  * nn_Al features: 1 hiddenlayer 'relu', with 50 epochs       
  * nn_Al Accuracy: 0.7310
  * nn_Al Loss: 0.5590

* AdaBoostClassifier Model Results:
  * nn_A2 features: 1 hiddenlayer 'relu', with 100 epochs      
  * nn_A2 Accuracy: 0.7287
  * nn_A2 Loss: 0.5660 

* DecisionTreeClassifier Model Results:
  * nn_A3 features- 2 hiddenlayer 'relu' + 1 hiddenlayer 'tanh' , with 50 epochs       
  * nn_A3 Accuracy: 0.7296
  * nn_A3 Loss: 0.5508 

---

## Technologies

The software operates on python 3.9 with the installation package imports embedded with Anaconda3 installation. Pandas, NumPy, hvplot, Matplotlib and scikitlearn are libraries for imported tools this program uses to build the program to analyze stock prices and set decision signals of when to buy and sell shares.  The application tools that you need for this module to construct models to perform ML algorithms are SVC, LogisticRegression, AdaBoostClassifier, DecisionTreeClassifier.  Please reference the formerly named ML algorithm classifiers on the official web site [Supervised Learning for scikit ML Classifiers Install Guide](https://scikit-learn.org/stable/supervised_learning.html) installation and documentation.   


---

## Installation Guide

Before running the applications open you terminal to install and check for your installations. First navigate to scikit-learn.org for installation instructions using the link below. Then verify if the installation as been completed. 

* (https://www.python.org/downloads/)

* [anaconda3](https://docs.anaconda.com/anaconda/install/windows/e) 

* [scikitlearn](https://scikit-learn.org/stable/install.html) 

```python libraries
pip install -U scikit-learn
python -m pip show scikit-learn      # to see which version of scikit-learn is installed
conda install numpy
```
```import pandas as pd
import numpy as np               # returns conditional classification values for signals
from pathlib import Path
import hvplot.pandas                                            # for chart visualations
import matplotlib.pyplot as plt
from sklearn import svm                                             # algorithm ML model
from sklearn.preprocessing import StandardScaler         # standardizes data for models from pandas.tseries.offsets import DateOffset        # allows length of time adjustments
from sklearn.metrics import classification_report 
from sklearn.linear_model import LogisticRegression                 # algorithm ML model
from sklearn.ensemble import AdaBoostClassifier                     # algorithm ML model
from sklearn.tree import DecisionTreeClassifier                     # algorithm ML model
```

---
# Usage

This application is launched from web-based Jupyter notebook utilizing Pandas and scikitlearn `StandardScaler` to preprocess data for categorical variables in the algorithm ML model computations. Scikit ML models SVC, LogisticRegression, AdaBoostClassifier, DecisionTreeClassifier allows tuning to adapt to different trading strategies and market environments.  and `accuracy` evaluation metrics.    

The program is developed in Jupyter notebook on a jupyter **.ipny** file. The Python library makes it possible to utilize pandas, numpy and pathlib to build this ML algorithm. The design applies the model-fit-predict process to make a binary classification of whether a startup is successful or not.
 

![NN Model Evals: Origin & A1](Images\Screenshot2022-06-15032835.png) 

![NN Model Evals: A2 & A3](Images\Screenshot2022-06-15033615.png) 



```python
machine_learning_trading_bot.ipynb
```
 

---

## Contributors

*Provided to you by digi-Borg FinTek*, 
Dana Hayes: nydane1@gmail.com

---

## License

Columbia U. Engineering