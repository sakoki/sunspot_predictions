# General functions for working with time series data


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