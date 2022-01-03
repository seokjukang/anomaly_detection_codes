import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as f
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tcn.parameters import Params


class TCN(nn.Module):
    def __init__(self):
        super(TCN, self).__init__()

        self.conv_1 = nn.Conv1d(1,
                                128,
                                kernel_size=tuple([2]),
                                dilation=tuple([1]),
                                padding=((2-1) * 1)
                                )
        self.conv_2 = nn.Conv1d(128,
                                128,
                                kernel_size=tuple([2]),
                                dilation=tuple([2]),
                                padding=((2-1) * 2)
                                )
        self.conv_3 = nn.Conv1d(128,
                                128,
                                kernel_size=tuple([2]),
                                dilation=tuple([4]),
                                padding=((2-1) * 4)
                                )
        self.conv_4 = nn.Conv1d(128,
                                128,
                                kernel_size=tuple([2]),
                                dilation=tuple([8]),
                                padding=((2-1) * 8)
                                )
        self.dense_1 = nn.Linear(31*128, 128)
        self.dense_2 = nn.Linear(128, Params.num_classes.value)

    def forward(self, x):
        x = self.conv_1(x)
        x = x[:, :, :-self.conv_1.padding[0]]
        x = f.relu(x)
        x = f.dropout(x, 0.05)

        x = self.conv_2(x)
        x = x[:, :, :-self.conv_2.padding[0]]
        x = f.relu(x)
        x = f.dropout(x, 0.05)

        x = self.conv_3(x)
        x = x[:, :, :-self.conv_3.padding[0]]
        x = f.relu(x)
        x = f.dropout(x, 0.05)

        x = self.conv_4(x)
        x = x[:, :, :-self.conv_4.padding[0]]
        x = f.relu(x)
        x = f.dropout(x, 0.05)

        x = x.view(-1, 31*128)
        x = f.relu(self.dense_1(x))
        x = self.dense_2(x)

        return f.log_softmax(x, dim=1)






