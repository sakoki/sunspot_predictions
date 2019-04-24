from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Set figure aesthetics
sns.set_style("darkgrid")


def univar_split_sequence(sequence, steps):
    """Accepts a sequence of data and splits it into samples to be inputs for the LSTM model

    The input is a array of length M

    The function splits the array into K samples, each with length equal to the number of specified steps
    Each resulting sample is shifted to the right by 1 index.

    :param sequence: univariable sequence data to be formatted
    :type sequence: numpy.ndarray
    :param int steps: number of 'X' points to be used as inputs to predict 'y'
    :return: 2 arrays, one for 'X' and one for 'y'
    """

    X, y = [], []
    for i in range(len(sequence)):
        # Find the end of this pattern
        end = i + steps
        # Check if end is out of bounds
        if end > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end], sequence[end]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


def multivar_split_sequence(sequence, steps):
    """Accepts sequences of data and splits it into samples to be inputs for the LSTM model

    The input is a matrix of size M x N, where M is the length and N is the number of sequences.
    Columns 1:N-1 are treated as 'X' and the Nth column is treated as 'y'

    The function splis the matrix into K samples, each with rows equal to the number of specified steps
    Each resulting sample is shifted by 1 row.

    :param sequence: multivariable sequence data to be formatted
    :type sequence: numpy.ndarray
    :param int steps: number of 'X' points to be used as inputs to predict 'y'
    :return: 2 arrays, one for 'X' and one for 'y'
    """

    X, y = [], []
    for i in range(len(sequence)):
        # Find the end of this pattern
        end = i + steps + 1
        # Check if end is out of bounds
        if end > len(sequence):
            break
        seq_x, seq_y = sequence[i:end-1, :-1], sequence[end-1, -1]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y)


def tensor_format(sequence, features):
    """Format input into 3D tensor by adding the features dimension

    The LSTM model accepts data with the following dimensions:
    [samples, timesteps, features]

    :param sequence: training data for LSTM
    :type sequence: numpy.ndarray
    :param int features: number of features
    """
    sequence = sequence.reshape((sequence.shape[0], sequence.shape[1], features))
    return sequence


def evaluation(true, predict, save=False, fname=None):
    """Calculate the evaluation score and plot predictions

    Calculate the Mean Squared Error and Root Mean Square Error.
    Graph predictions as line plot

    :param list true: true values of time series
    :param list predict: predicted values of time series
    :param bool save: save figure as png (default False)
    :param str fname: name of image file (default None)
    """

    MSE = mean_squared_error(true, predict)
    RMSE = np.sqrt(MSE)
    print("MSE: {:.4f}".format(MSE))
    print("RMSE: {:.4f}".format(RMSE))
    # Plot true values and predicted values
    plt.figure(figsize=(20, 10))
    plt.plot(true)
    plt.plot(predict)
    plt.legend(['true', 'prediction'], loc='upper left')
    plt.title("LSTM", fontdict={'fontsize': 30})
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Time step', fontsize=20)
    plt.ylabel('Sunspots', fontsize=20)
    if save:
        plt.savefig('./{}'.format(fname))
    plt.show()


def plot_lstm(history, fname=None, val=True):
    """Display history of LSTM training

    :param str fname: name of image file (default None)
    :param bool val: plot validation error (default True)
    """

    rmse = np.sqrt(history.history['loss'])
    plt.figure(figsize=(15, 9))
    plt.plot(rmse)
    if val:
        val_rmse = np.sqrt(history.history['val_loss'])
        plt.plot(val_rmse)
        plt.legend(['train', 'validation'], loc='upper left')
    plt.title('model RMSE', fontdict={'fontsize': 20})
    plt.ylabel('rmse', fontdict={"fontsize": 18})
    plt.xlabel('epochs', fontdict={"fontsize": 18})
    plt.savefig('./{}_lstm_loss.png'.format(fname))
    plt.show()
