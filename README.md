# Stock Forecaster

**Next Day Stock Price Predictor - Neural Network (Machine Learning) Model**

Machine learning Python project that leverages the neural network capabilities of the Tensorflow and Keras packages. It formats historic stock prices into time series data to train a long short-term memory model (LSTM) and predicts the next day's stock price, providing graphical prediction analysis.

## Techniques Used for Training

- Utilizes the "sliding window" technique to prepare stock price data as input for the machine learning model (20 day sliding window).
- Learns from the first 90% of the actual historic stock movement as training data, then tests its accuracy (validation split) on the last 10% of the actual historic stock movement.
- Training RMSE is based on the first 90% of the historic stock price data, and testing RMSE is based on the last 10% of the historic stock price data.

## Program Features

- Predicts the next day of trading.
- Utilizes 50 neurons for fitting with a 20% dropout layer.
- Incorporates early stopping technique with 15 patience to prevent overfitting during training and testing.
- Random seeds for NumPy and Tensorflow packages are stabilized, ensuring reproducible outputs.

## Program Limitations

- Univariate; the program only considers stock movement rather than market news, so business deals and other market factors cannot be considered.
- Any other data requires slight reformatting to be accepted by the program (i.e., unnecessary formatting must be removed, leaving only column titles).
- Running on Python 3.11.5 due to library packages compatability
