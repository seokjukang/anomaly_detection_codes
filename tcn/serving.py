import sys
from random import randint

import numpy as np
import torch
from _common.config import Config
from tcn.tcn_model import TCN
from tcn.util import Util


def serve(model, x, config):
    print(f"input x: {x[:, :, 1:2]}")
    print(f"input x.shape: {x.shape}")

    with torch.no_grad():
        if config.isCudaAvailable:
            x = torch.Tensor(x).cuda().float()
        else:
            x = torch.Tensor(x).float()

        x = x.to(config.device)
        outputs = model(x)
        _, predicted = torch.max(outputs.data, 1)
        detatched_pred = predicted.detach().cpu().numpy()
        print(f"predicted: {predicted}")
        print(f"detatched_pred: {detatched_pred[0]}")

        return detatched_pred


# serve test
x_train, x_test, y_train, y_test = None, None, None, None
util = Util()
try:
    PATH = "tcn_model.ckpt"
    config = Config()
    device = config.device

    # set model
    model = TCN()
    model.load_state_dict(torch.load(PATH))
    model.to(device)
    model.eval()

    print(f"device: {device}")
    print(f"model: {model}")

    x_train, x_test, y_train, y_test = util.get_train_test_data('../_data/creditcard.csv')
    for i in range(100):
        x = x_test[randint(1, len(x_test) - 1), :]
        x = np.array(x).reshape((x.shape[0], 1, x.shape[1]))
        result = serve(model, x, config)
        if result == 0:
            print("result: normal\n")
        else:
            print("result: >>> anomaly <<<\n")

except Exception as e:
    print(e)
    sys.exit(1)


