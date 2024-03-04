# Stock Forcaster
Next Day Stock Price Predictor - Neural Network (Machine Learning) Model

Machine learning Python project that takes advantage of neural network capabilities of the Tensorflow and Keras packages. Formats historic stock prices into time series data to train long short-term memory model (LSTM). Predicts the next day stock price and outputs graphical prediciton analysis.

Techniques used for training

Utilizes "sliding window" technique which prepares stock price data to be used as input for the machine learning model (20 day sliding window)
Learns from the first 90% of the actual historic stock movement as training data then tests its accuracy--validation split--on the last 10% of the actual historic stock movement
Training RMSE is based on the first 90% of the historic stock price data and the testing RMSE is based on the last 10% of the historic stock price data
Program features

Predicts the next day of trading 
Utilizes 50 neurons for fitting with a 20% dropout layer 
Incorporates early stopping technique with 15 patience to prevent overfitting during training and testing
Random seeds for NumPy and Tensorflow packages are stabilized and thus outputs are reproducable
Program limitations

Univariate; program only considers stock movement rather than market news. Therefore, business deals and other market factors cannot be considered in this program
Any other data requires slight amounts of reformatting in order to be accepted by the program (i.e. unnecessary formatting must be removed other than column titles)
