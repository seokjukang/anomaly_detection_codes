import math

import numpy as np
import pandas as pd
from keras.losses import mean_squared_error


class LSTMPredict:
    def __init__(self):
        print("")

    def serve(self, model, df, X):
        testing_pred = model.predict(x=X)
        print("testing_pred: ", testing_pred)

        testing_dataset = X.reshape(
            (X.shape[0] * X.shape[1]),
            X.shape[2]
        )
        print("testing_dataset.shape: ", testing_dataset.shape)

        testing_pred = testing_pred.reshape(
            (testing_pred.shape[0] * testing_pred.shape[1]),
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

        z = zip(dist >= threshold, dist)

        y_label = []
        error = []
        for idx, (is_anomaly, dist) in enumerate(z):
            if is_anomaly:
                y_label.append(1)
            else:
                y_label.append(0)
            error.append(dist)

        adf = pd.DataFrame({
            'Datetime': df['Datetime'],
            'observation': df['value'],
            'error': error,
            'anomaly': y_label
        })
        len(adf[adf['anomaly'] == 1])
        print(adf.head(5))

        len(adf[adf['anomaly'] == 1])
        print(adf.head(5))

        anomaliesDF = adf.query('anomaly == 1')

        return anomaliesDF