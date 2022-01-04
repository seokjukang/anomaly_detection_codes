import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Util:
    def get_train_test_data(self, file_path):
        try:
            # read file
            df = pd.read_csv(filepath_or_buffer=file_path, header=0, sep=',')
            print(df.shape[0])
            df.head()

            # scaling & sampling
            df['Amount'] = StandardScaler().fit_transform(
                df['Amount'].values.reshape(-1, 1)
            )
            df['Time'] = StandardScaler().fit_transform(
                df['Time'].values.reshape(-1, 1)
            )
            df_0 = df.query('Class == 0').sample(20000)
            df_1 = df.query('Class == 1').sample(400)
            df = pd.concat([df_0, df_1])
            df = df.sort_values(by=['Time'], axis=0)
            df.head()
            print(f"shape of dataframe: {df.shape[0], df.shape[1]}")
            print(f"normal count: {len(df.query('Class == 0'))}")
            print(f"anomaly count: {len(df.query('Class == 1'))}")

            x_train, x_test, y_train, y_test = train_test_split(
                df.drop(labels=['Time', 'Class'], axis=1),
                df['Class'],
                test_size=0.2,
                random_state=42
            )
            print(x_train.shape, 'train samples')
            print(x_test.shape, 'test samples')

            return x_train, x_test, y_train, y_test
        except Exception as e:
            print(e)
            return None
