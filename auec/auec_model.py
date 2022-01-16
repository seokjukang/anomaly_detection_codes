from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

base_log_file_name = "autoencoder"


class Autoencoder:
    def __init__(self):
        self.log_file_name = base_log_file_name

    def get_deep_model(self, x_train):
        self.log_file_name = "deep_" + base_log_file_name

        try:
            encoding_dim = 16
            input_dim = x_train.shape[1]  # 29

            input_arr = Input(shape=(input_dim,))
            encoded = Dense(encoding_dim, activation='relu')(input_arr)
            encoded = Dense(8, activation='relu')(encoded)
            encoded = Dense(4, activation='relu')(encoded)

            decoded = Dense(8, activation='relu')(encoded)
            decoded = Dense(encoding_dim, activation='relu')(decoded)
            decoded = Dense(input_dim, activation='softmax')(decoded)

            autoencoder = Model(input_arr, decoded)
            autoencoder.summary()

            return autoencoder
        except Exception as e:
            print(e)
            return None

    def get_basic_mdl(self, x_train):
        self.log_file_name = "simple_" + base_log_file_name

        try:
            encoding_dim = 12
            input_dim = x_train.shape[1]

            input_arr = Input(shape=(input_dim,))
            encoded = Dense(encoding_dim, activation='relu')(input_arr)
            decoded = Dense(input_dim, activation='softmax')(encoded)

            autoencoder = Model(input_arr, decoded)
            autoencoder.summary()
            return autoencoder
        except Exception as e:
            print(e)
            return None

    def get_sparse_mdl(self, x_train):
        self.log_file_name = "sparse_" + base_log_file_name

        try:
            encoding_dim = 12
            input_dim = x_train.shape[1]

            input_arr = Input(shape=(input_dim,))
            encoded = Dense(encoding_dim, activation='relu',
                            activity_regularizer=regularizers.l1(10e-5))(input_arr)
            decoded = Dense(input_dim, activation='softmax')(encoded)

            autoencoder = Model(input_arr, decoded)
            autoencoder.summary()
            return autoencoder
        except Exception as e:
            print(e)
            return None

