import sys
import numpy as np
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn

from _common.config import Config
from tcn.parameters import Params
from tcn.tcn_model import TCN
from tcn.util import Util


def train(model, device, x_train, y_train, criterion, optimizer, epoch, save_dir='tcn_model.ckpt'):
    try:
        if config.isCudaAvailable:
            x_train = torch.Tensor(x_train).cuda().float()
            y_train = torch.Tensor(y_train).cuda().long()
        else:
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

        if config.isCudaAvailable:
            x_test = torch.Tensor(x_test).cuda().float()
            y_test = torch.Tensor(y_test).cuda().long()
        else:
            x_test = torch.Tensor(x_test).float()
            y_test = torch.Tensor(y_test).long()

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


# train
x_train, x_test, y_train, y_test = None, None, None, None
device = None
util = Util()
try:
    config = Config()
    device = config.device
    if device is None:
        sys.exit(1)
    print(f"current device : {device}")

    model = TCN().to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Params.learning_rate.value)
    x_train, x_test, y_train, y_test = util.get_train_test_data('../_data/creditcard.csv')

    for epoch in range(0, Params.num_epochs.value):
        train(model, device, x_train, y_train, loss, optimizer, epoch)
except Exception as e:
    print(e)

# test model
try:
    PATH = "tcn_model.ckpt"
    model_load = TCN()
    model_load.load_state_dict(torch.load(PATH))
    model_load.to(device)
    model_load.eval()

    test(model_load, device, x_test, y_test)
except Exception as e:
    print(e)

