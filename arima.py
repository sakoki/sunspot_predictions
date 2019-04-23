from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Custom function
from decorators import timer

# import statsmodels

# Set figure aesthetics
sns.set_style("darkgrid")

def time_series_split(data, split):
    """Split time series data into train and test set

    :param DataFrame data: data frame with n x time points and m x variables
    :param float split: proportion of train and test split
    :return: split train and test data
    :rtype: DataFrame
    """

    split_idx = int(data.shape[0] * split)
    train = data.iloc[:split_idx, :]
    test = data.iloc[split_idx:, :]

    print("Size of original data: {}".format(data.shape[0]))
    print("Size of training data: {}".format(train.shape[0]))
    print("Size of test data: {}".format(test.shape[0]))

    return train, test


def ARIMA_forecast(train, P, D, Q):
    """Train a ARIMA model with specified parameters

    :param train: training data
    :type train: list of float
    :param int P: number of lag observations
    :param int D: degree of differencing
    :param int Q: size of the moving average window
    :return: predicted values
    :rtype: float
    """
    model = sm.tsa.ARIMA(train, order=(P, D, Q))
    # Display training results in console
    model_fit = model.fit(disp=1)
    # Get the next prediction for the next time point
    prediction = model_fit.forecast()[0][0]
    return prediction


def ARIMA_describe(model):
    """Provide descriptive information about the trained model

    :param model: trained ARIMA model
    :type model: statsmodels.tsa.arima_model.ARIMA
    :param int P: number of lag observations
    :param int D: degree of differencing
    :param int Q: size of the moving average window
    """

    plt.figure(figsize=(12,7))
    # Display model summary
    print(model.summary())
    # Plot residual errors as line plot and density
    residuals = pd.DataFrame(model.resid)
    residuals.plot()
    residuals.plot(kind='kde')


def evaluation(true, predict, P, D, Q):
    """Calculate the evaluation score and plot predictions

    Calculate the Mean Squared Error and Root Mean Square Error.
    Graph predictions as line plot

    :param true: true values of time series
    :param predict: predicted values of time series
    """

    MSE = mean_squared_error(true, predict)
    RMSE = np.sqrt(MSE)
    print("MSE: {:.4f}".format(MSE))
    print("RMSE: {:.4f}".format(RMSE))
    # Plot true values and predicted values
    plt.figure(figsize=(20, 10))
    plt.plot(true)
    plt.plot(predict)
    plt.title("ARIMA - (P:{}, D:{}, Q:{})".format(P, D, Q), fontdict={'fontsize': 30})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Sunspots', fontsize=20)
    plt.show()


def visualize(data, P, D, Q):
    """Visualize time series data as a lineplot"""
    plt.figure(figsize=(20, 10))
    plt.plot(data)
    plt.title("ARIMA - (P:{}, D:{}, Q:{})".format(P, D, Q), fontdict={'fontsize': 30})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Time', fontsize=20)
    plt.ylabel('Sunspots', fontsize=20)
    plt.show()


def rolling_predictions(data, y, P, D, Q, split):
    """Generate rolling predictions based on true values as they are observed

    For rolling predictions the test data is a split of the original data. Since
    the true values are known, we can evaluate its accuracy.

    :param DataFrame data: time series data
    :param str y: specify column that contains values to predict
    :param int P: number of lag observations
    :param int D: degree of differencing
    :param int Q: size of the moving average window
    :param float split: proportion of train and test split (default None)
    :return: predicted values
    """

    # Generate train and test data
    train, test = time_series_split(data, split)
    train_list = train[y].tolist()
    test_list = test[y].tolist()

    predictions = []

    for time in range(len(test_list)):
        # Get true value
        actual_value = test_list[time]
        # Forecast value
        predicted_value = ARIMA_forecast(train_list, P, D, Q)
        predictions.append(predicted_value)
        # Extend training data with true values
        train_list.append(actual_value)

    evaluation(test_list, predictions, P, D, Q)
    return predictions


def future_predictions(data, y, P, D, Q, N):
    """Generate predictions based on previous predictions

    :param DataFrame data: time series data
    :param str y: specify column that contains values to predict
    :param int P: number of lag observations
    :param int D: degree of differencing
    :param int Q: size of the moving average window
    :param int N: specify number of future predictions to make
    :return: predicted values
    """

    print("Forecasting {} time steps into the future".format(N))
    train_list = data[y].tolist()

    predictions = []

    for time in range(N):
        # Forecast value
        predicted_value = ARIMA_forecast(train_list, P, D, Q)
        predictions.append(predicted_value)
        # Add prediction to training data
        train_list.append(predicted_value)

    visualize(predictions, P, D, Q)

    return predictions


@timer
def ARIMA_model(data, y, P, D, Q, split=None, future=False, N=None, output=False):
    """Generate time series predictions with ARIMA model with specified parameters

    This model can be configured to make rolling predictions (true values appended to training data
    as new observations after each iteration), or make predictions based on previous predictions.

    :param DataFrame data: time series data
    :param str y: specify column that contains values to predict
    :param int P: number of lag observations
    :param int D: degree of differencing
    :param int Q: size of the moving average window
    :param float split: proportion of train and test split (default None)
    :param bool future: make future predictions using prior predictions (default False)
    :param int N: if future is True, specify number of predictions to make (default None)
    :param bool output: specify whether to return output or not (default False)
    :return: predicted values
    """

    if not future:
        predictions = rolling_predictions(data, y, P, D, Q, split)
    else:
        predictions = future_predictions(data, y, P, D, Q, N)

    if output:
        return predictions
    return

