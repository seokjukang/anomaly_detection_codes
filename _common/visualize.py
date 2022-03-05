import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


class Visualization:
    labels = ["Normal", "Anomaly"]

    def draw_confusion_matrix(self, y, ypred):
        matrix = confusion_matrix(y, ypred)

        plt.figure(figsize=(10, 8))
        colors = ["orange", "green"]
        sns.heatmap(matrix, xticklabels=self.labels, yticklabels=self.labels,
                    cmap=colors, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

    def draw_anomaly(self, y, error, threshold):
        groupDF = pd.DataFrame({
            'error': error,
            'true': y
        }).groupby('true')

        figure, axes = plt.subplots(figsize=(12, 8))

        for name, group in groupDF:
            axes.plot(group.index, group.error,
                      marker='x' if name == 1 else 'o',
                      linestyle='', color='r' if name == 1 else 'g',
                      label="Anomaly" if name == 1 else "Normal")

        axes.hlines(threshold,
                    axes.get_xlim()[0],
                    axes.get_xlim()[1],
                    colors="b",
                    zorder=100,
                    label='Threshold'
                    )
        axes.legend()

        plt.title("Anomalies")
        plt.ylabel("Error")
        plt.xlabel("Data")
        plt.show()

    def draw_error(self, error, threshold):
        plt.plot(error,
                 maker='o',
                 ms=3.5,
                 linestyle='',
                 label='Point'
                 )
        plt.hlines(threshold,
                   xmin=0,
                   xmax=len(error) - 1,
                   colors="b",
                   zorder=100,
                   label='Threshold'
                   )
        plt.legend()
        plt.title("Reconstruction error")
        plt.ylabel("Error")
        plt.xlabel("Data")
        plt.show()


