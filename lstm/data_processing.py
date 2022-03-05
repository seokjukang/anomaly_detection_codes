import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class LSTMDataProcessing:
    def __init__(self):
        print("")

    def get_data(self, file_path, log_file_name):
        log_name = log_file_name
        data_file_path = file_path
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

        scaler = MinMaxScaler(feature_range=(0, 1))
        df['scaled_value'] = pd.DataFrame(
            scaler.fit_transform(
                pd.DataFrame(df['value'])
            ),
            columns=['value']
        )
        print('scaled df.shape: ', df.shape)
        print('scaled df.head: ')
        print(df.head(5))

        sequence = np.array(df['scaled_value'])
        print("sequence", sequence)

        time_steps = 48
        samples = len(sequence)
        print("samples", samples)

        trim = samples % time_steps
        print("trim: ", trim)
        subsequences = int(samples / time_steps)
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

        return training_dataset, testing_dataset