import sys

import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import RMSprop, Adam, Nadam
from sklearn.metrics import classification_report, roc_auc_score

from auec.auec_model import Autoencoder
from auec.parameters import Params
from auec.util import Util
from auec.visualize import Visualization

PATH = "autoencoder_model.h5"


def train(model,
          x_train, x_test,
          cb,
          optimizer=RMSprop(),
          loss='mean_squared_error',
          metrics=['mae', 'accuracy'],
          batch_size=Params.batch_size.value,
          epochs=Params.epochs.value):
    try:
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
        history = model.fit(
            x_train,
            x_train,
            batch_size=Params.batch_size.value,
            epochs=Params.epochs.value,
            verbose=1,
            shuffle=True,
            validation_data=(x_test, x_test),
            callbacks=[cb]
        )

        return history
    except Exception as e:
        print(e)
        return None, None


def test(model, x_test, y_test, threshold=Params.threshold.value):
    score = model.evaluate(x_test, x_test, verbose=1)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])

    y_pred = model.predict(x_test)
    y_dist = np.linalg.norm(x_test - y_pred, axis=-1)
    z = zip(y_dist >= threshold, y_dist)
    y_label = []
    error = []

    for idx, (is_anomaly, y_dist) in enumerate(z):
        if is_anomaly:
            y_label.append(1)
        else:
            y_label.append(0)
        error.append(y_dist)

    print(classification_report(y_test, y_label))
    roc_auc_score(y_test, y_label)

    return y_label, error


def save_model(mdl, path):
    mdl.save(path)


# train
model = None
callback = None
encoder = Autoencoder()
util = Util()
x_train, x_test, y_train, y_test = \
    util.get_train_test_data('../_data/creditcard.csv')
if x_train is not None:
    model = encoder.get_deep_model(x_train)
    callback = TensorBoard(log_dir='../logs/{0}'.format(encoder.log_file_name))

hist = train(model, x_train, x_test, callback)

# save model
save_model(model, PATH)

# test
y_lbl, err = test(model, x_test, y_test)

# visualize
viz = Visualization()
viz.draw_confusion_matrix(y_test, y_lbl)
viz.draw_anomaly(y_test, err, Params.threshold.value)


