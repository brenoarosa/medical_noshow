"""
Script responsable to train a model
"""

import joblib

import torch as pt
from torch.nn.functional import (binary_cross_entropy_with_logits, mse_loss,
                                 binary_cross_entropy)

from utils.training import train_model

from mlp import MLP

X_train = joblib.load("preprocessed_data/X_train.p")
X_val = joblib.load("preprocessed_data/X_val.p")
X_test = joblib.load("preprocessed_data/X_test.p")
y_train = joblib.load("preprocessed_data/y_train.p")
y_val = joblib.load("preprocessed_data/y_val.p")
y_test = joblib.load("preprocessed_data/y_test.p")
print("{} training samples.".format(X_train.shape[0]))
print("{} validation samples.".format(X_val.shape[0]))
print("{} test samples.".format(X_test.shape[0]))


X_train = pt.from_numpy(X_train.astype("float32"))
y_train = pt.from_numpy(y_train.astype("float32"))
X_val = pt.from_numpy(X_val.astype("float32"))
y_val = pt.from_numpy(y_val.astype("float32"))


tensors = [(X_train, y_train), (X_val, y_val)]
loss_func = binary_cross_entropy

opt = {
    "batch_size": 4096,
    "epochs": 140,
    "early_stop": {
        "monitor": "train_loss",
        "mode": "min",
        "patience": 5,
        "min_delta": .0003
    }
}

sizes = {11: [X_train.shape[1], 100, 50, 15, 1]}

for i, layer_sizes in sizes.items():
    opt["model_name"] = "model_{}".format(i)
    model = MLP(layer_sizes)
    optimizer = pt.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    train_model(model, tensors, optimizer, loss_func, **opt)
