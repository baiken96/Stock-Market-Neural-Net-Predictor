# Stock-Market-Neural-Net-Predictor

This project demonstrates the use of a LSTM neural network to predict stock market performance for a specified number of business days in the future. The virtual environment settings are included, so this repository can be opened as a Pycharm project for convenience.

The command to evaluate the performance of the model is:
python testsuite.py fulldataset.csv goldstandard.csv (optional: "load")

To evaluate the performance of the model, run the testsuite.py script with the full dataset csv file as arg1, and the gold standard csv file (the closing prices for the next 10 days) as arg2. The full dataset will be partitioned into training and validation sets automatically. The script will print to a file ([stock name]_results.txt) its mean squared error (MSE) in validation, and prediction MSE for 1, 5, and 10 days in the future. To load a pre-trained model, include “load” as arg3.

The command to make a prediction is:
python predict.py modelprefix (optional: #_days)

To make a prediction from a pre-trained model, run the predict.py script with arg1 as the prefix of the model (should be the stock market abbreviation for the company) and optional arg2 as the number of days ahead to predict (by default it will print separate predictions for 1, 5, and 10).

This code was based on a tutorial available at https://www.analyticsvidhya.com/blog/2018/10/predicting-stock-price-machine-learningnd-deep-learning-techniques-python/

All data was obtained from https://www.macrotrends.net/

All code was tested with Python 3.7.
