import sys
import numpy as np
import pandas as pd
import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as f

# Hyper-parameters
from tcn.parameters import Params
from tcn.tcn_model import TCN


# num_epochs = 30
# num_classes = 2
# learning_rate = 0.002


def get_device():
    try:
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    except Exception as e:
        return None


def get_train_test_data(file_path):
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

        # reshape the train and test data sets
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


def train(model, device, x_train, y_train, criterion, optimizer, epoch, save_dir='tcn_model.ckpt'):
    try:
        # x_train = torch.Tensor(x_train).cuda().float()
        # y_train = torch.Tensor(y_train).cuda().long()
        x_train = torch.Tensor(x_train).float()
        y_train = torch.Tensor(y_train).long()

        x_train.to(device)
        y_train.to(device)

        # forward pass
        outputs = model(x_train)
        loss = criterion(outputs, y_train.squeeze(1))

        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch {}/{}, Loss: {:.4f}'.format(epoch + 1,
                                                 Params.num_epochs.value,
                                                 loss.item()
                                                 )
              )

        torch.save(model.state_dict(), save_dir)
    except Exception as e:
        print(f"error: {e}")


def test(model, device, x_test, y_test):
    preds = []
    y_true = []

    # set model to evaluation mode
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0

        x_test = torch.Tensor(x_test).cuda().float()
        y_test = torch.Tensor(y_test).cuda().long()

        x_test = x_test.to(device)
        y_test = y_test.to(device)
        y_test = y_test.squeeze(1)

        outputs = model(x_test)
        _, predicted = torch.max(outputs.data, 1)

        total += y_test.size(0)
        correct += (predicted == y_test).sum().item()

        detatched_pred = predicted.detach().cpu().numpy()
        detatched_label = y_test.detach().cpu().numpy()

        for i in range(0, len(detatched_label)):
            preds.append(detatched_pred[i])
            y_true.append(detatched_label[i])

        print('Test Accuracy of the model on the test images: {:.2%}'.format(correct / total))

        preds = np.eye(Params.num_classes.value)[preds]
        y_true = np.eye(Params.num_classes.value)[y_true]
        auc = roc_auc_score(np.round(preds), y_true)
        print("AUC: {:.2%}".format(auc))


# check device
device = get_device()
if device is None:
    sys.exit(1)
print(f"current device : {device}")

model = TCN().to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Params.learning_rate.value)

# get data
x_train, x_test, y_train, y_test = None, None, None, None
try:
    x_train, x_test, y_train, y_test = get_train_test_data('data/creditcard.csv')
except Exception as e:
    print(e)

# train
for epoch in range(0, Params.num_epochs.value):
    train(model, device, x_train, y_train, loss, optimizer, epoch)
