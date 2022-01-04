import numpy as np
from keras.models import load_model

from auec.parameters import Params
from auec.util import Util


def serve(model, x_test, threshold=Params.threshold.value):
    score = model.evaluate(x_test, x_test, verbose=1)
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])

    y_pred = model.predict(x_test)
    y_dist = np.linalg.norm(x_test - y_pred, axis=-1)
    z = zip(y_dist >= threshold, y_dist)
    y_label = []
    errors = []

    for idx, (is_anomaly, y_dist) in enumerate(z):
        if is_anomaly:
            y_label.append(1)
        else:
            y_label.append(0)
        errors.append(y_dist)

    return y_label, errors


def load(path):
    try:
        return load_model(path)
    except Exception as e:
        print(e)
        return None


# serve test
util = Util()
x_train, x_test, y_train, y_test = util.get_train_test_data('../_data/creditcard.csv')
model = load('autoencoder_model.h5')
if model is not None:
    result, errors = serve(model, x_test)
    anomalies = []
    errors_over_thres = []
    for v in result:
        if v == 1:
            anomalies.append(v)
    for err in errors:
        if err >= Params.threshold.value:
            errors_over_thres.append(err)

    print(f"anomalies count: {len(anomalies)}")
    print(f"errors_over_thres count: {len(errors_over_thres)}")
