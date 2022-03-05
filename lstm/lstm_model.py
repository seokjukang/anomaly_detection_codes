from keras.callbacks import TensorBoard
from keras.layers import Dense, LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

from _common.visualize import Visualization

base_log_file_name = "lstm"


class LSTMModel:
    def __init__(self):
        self.log_file_name = base_log_file_name

    def get_basic_model(self) -> Sequential:
        self.log_file_name = "basic_" + base_log_file_name

        time_steps = 48
        metric = 'mean_absolute_error'

        try:
            seq_model = Sequential()
            seq_model.add(
                LSTM(
                    units=32,
                    activation='tanh',
                    input_shape=(time_steps, 1),
                    return_sequences=True
                )
            )
            seq_model.add(
                Dense(1, activation='sigmoid')
            )
            seq_model.compile(
                optimizer='adam',
                loss='mean_absolute_error',
                metrics=[metric]
            )
            # print(model.summary())

            return seq_model
        except Exception as e:
            print(e)
            return None


if __name__ == "__main__":
    log_name = "nyc_taxi"
    data_file_path = "../_data/nyc_taxi.csv"
    print(f"log_name: {log_name}")
    print(f"data_file_path: {data_file_path}")

    df = pd.read_csv(
        filepath_or_buffer=data_file_path,
        header=0,
        sep=','
    )
    print('df.shape', df.shape)
    print('df.head: ')
    print(df.head(5))

    df['Datetime'] = pd.to_datetime(df['timestamp'])
    print('df.head: ')
    print(df.head(3))
    # df.shape

    # df.plot(x='Datetime', y='value', figsize=(15, 6))
    # plt.xlabel('Date time')
    # plt.ylabel('Value')
    # plt.title('Time Series of value by date time')

    # df.value.describe()
    # df.Datetime.describe()

    # fig, (ax1) = plt.subplots(ncols=1, figsize=(8, 5))
    # ax1.set_title('Before Scaling')
    # sns.kdeplot(df['value'], ax=ax1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    df['scaled_value'] = pd.DataFrame(
        scaler.fit_transform(
            pd.DataFrame(df['value'])
        ),
        columns=['value']
    )
    print('scaled df.shape: ', df.shape)
    print('scaled df.head: ')
    df.head(5)

    # fig1, (ax2) = plt.subplots(ncols=1, figsize=(8, 5))
    # ax2.set_title('After Scaling')
    # sns.kdeplot(df['scaled_value'], ax=ax2)

    sequence = np.array(df['scaled_value'])
    print("sequence", sequence)

    time_steps = 48
    samples = len(sequence)
    print("samples", samples)

    trim = samples % time_steps
    print("trim: ", trim)
    subsequences = int(samples/time_steps)
    print("subsequences", subsequences)

    sequence_trimmed = sequence[:samples - trim]
    sequence_trimmed.shape = (
        subsequences, time_steps, 1
    )
    print("sequence_trimmed.shape: ", sequence_trimmed.shape)

    training_dataset = sequence_trimmed
    print("train dataset: ", training_dataset)



    testing_dataset = sequence_trimmed
    print("testing_dataset.shape: ", testing_dataset.shape)




    testing_pred = model.predict(x=testing_dataset)
    print("testing_pred: ", testing_pred)

    testing_dataset = testing_dataset.reshape(
        (testing_dataset.shape[0]*testing_dataset.shape[1]),
        testing_dataset.shape[2]
    )
    print("testing_dataset.shape: ", testing_dataset.shape)

    testing_pred = testing_pred.reshape(
        (testing_pred.shape[0]*testing_pred.shape[1]),
        testing_pred.shape[2]
    )
    print("testing_pred.shape; ", testing_pred.shape)

    errorDF = testing_dataset - testing_pred
    print("errorDF.shape: ", errorDF.shape)

    rmse = math.sqrt(mean_squared_error(testing_dataset, testing_pred))
    print("Test RMSE: %.3f" % rmse)

    dist = np.linalg.norm(
        testing_dataset - testing_pred, axis=1
    )
    scores = dist.copy()
    print("scores.shape: ", scores.shape)

    scores.sort()
    cutoff = int(0.999 * len(scores))
    print("cutoff: ", cutoff)
    print("cutoff scores: ", scores[cutoff:])

    threshold = scores[cutoff]
    print("threshold: ", threshold)

    plt.figure(figsize=(24, 16))
    plt.plot(testing_dataset, color='green')
    plt.plot(testing_pred, color='red')

    z = zip(dist >= threshold, dist)

    y_label = []
    error = []
    for idx, (is_anomaly, dist) in enumerate(z):
        if is_anomaly:
            y_label.append(1)
        else:
            y_label.append(0)
        error.append(dist)

    # viz = Visualization()
    # viz.draw_anomaly(y_label, error, threshold) # anomaly, diff between real and predicted value, threshold
    # viz.draw_error(error, threshold)

    adf = pd.DataFrame({
        'Datetime': df['Datetime'],
        'observation': df['value'],
        'error': error,
        'anomaly': y_label
    })
    len(adf[adf['anomaly'] == 1])
    print(adf.head(5))

    figure, axes = plt.subplots(
        figsize=(12, 6)
    )
    axes.plot(adf['Datetime'],
              adf['observation'],
              color='g')
    anomaliesDF = adf.query('anomaly == 1')
    axes.scatter(
        anomaliesDF['Datetime'].values,
        anomaliesDF['observation'],
        color='r'
    )
    plt.xlabel('Date time')
    plt.ylabel('observation')
    plt.title('Time Series of value by date time')
    plt.show()






