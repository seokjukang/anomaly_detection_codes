import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Util:
    def get_train_test_data(self, file_path):
        x_train, x_test, y_train, y_test = None, None, None, None
        try:
            df = pd.read_csv(file_path)
            if df is None or len(df) == 0:
                raise Exception

            print(df.head(5))
            print(df.shape)

            df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
            df['Time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))

            print(df.head(5))
            print(df.shape)

            anomalies = df[df["Class"] == 1]
            normal = df[df["Class"] == 0]
            print(f"anomalies.shape: {anomalies.shape}")
            print(f"normal.shape: {normal.shape}")

            for i in range(0, 20):
                normal = normal.iloc[np.random.permutation(len(normal))]
            # print(f"randomly permuted normal: {normal.head(5)}")

            data_set = pd.concat([normal[:10000], anomalies])
            data_set = data_set.iloc[np.random.permutation(len(data_set))]
            print(f"data_set(concated normal + anomalies) shape: {data_set.shape}")

            # x_train, x_test
            x_train, x_test = train_test_split(data_set, test_size=0.4, random_state=42)
            x_train = x_train.sort_values(by=['Time'])
            x_test = x_test.sort_values(by=['Time'])

            # y_train, y_test
            y_train = x_train["Class"]
            y_test = x_test["Class"]

            # reshape the train and test _data sets
            x_train = np.array(x_train).reshape((x_train.shape[0], 1, x_train.shape[1]))
            x_test = np.array(x_test).reshape((x_test.shape[0], 1, x_test.shape[1]))
            print(f"x_train reshaped: {x_train.shape}")
            print(f"x_test reshaped: {x_test.shape}")

            y_train = np.array(y_train).reshape((y_train.shape[0], 1))
            y_test = np.array(y_test).reshape((y_test.shape[0], 1))
            print(f"y_train reshaped: {y_train.shape}")
            print(f"y_test reshaped: {y_test.shape}")
        except Exception as e:
            print(e)

        return x_train, x_test, y_train, y_test
