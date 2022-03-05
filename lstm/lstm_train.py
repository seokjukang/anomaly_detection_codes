from keras.callbacks import TensorBoard


class LSTMTrain:
    def __init__(self):
        print("")
        self.log_name = None

    def train(self, model, X, Y, log_file_name):
        log_name = log_file_name
        batch_size = 32
        epochs = 20

        lstm = model
        model = lstm.get_basic_model()
        print(model.summary())

        hist = model.fit(
            x=X,
            y=X,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X, X),
            callbacks=[
                TensorBoard(log_dir='../logs/{0}'.format(log_name))
            ]
        )