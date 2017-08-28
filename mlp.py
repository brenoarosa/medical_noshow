from typing import Iterable

import numpy as np
import joblib
from copy import deepcopy

import torch as pt
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import (binary_cross_entropy_with_logits, mse_loss,
                                binary_cross_entropy)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

from utils.tensorboard import TBLogger
from utils.class_weights import compute_sample_weight

workers = joblib.cpu_count()

log_path = "./log/dev"
logger = TBLogger(log_path)

batch_size = 4096
num_classes = 10
epochs = 20
eps=1e-08


X_train = joblib.load("preprocessed_data/X_train.p")
X_val = joblib.load("preprocessed_data/X_val.p")
X_test = joblib.load("preprocessed_data/X_test.p")
y_train = joblib.load("preprocessed_data/y_train.p")
y_val = joblib.load("preprocessed_data/y_val.p")
y_test = joblib.load("preprocessed_data/y_test.p")
print("{} training samples.".format(X_train.shape[0]))
print("{} test samples.".format(X_test.shape[0]))
print("{} validation samples.".format(X_val.shape[0]))


layer_sizes = [X_train.shape[1], 500, 300, 100, 1]

X_train = pt.from_numpy(X_train.astype("float32"))
y_train = pt.from_numpy(y_train.astype("float32"))
X_test = pt.from_numpy(X_test.astype("float32"))
y_test = pt.from_numpy(y_test.astype("float32"))

train_dataset = TensorDataset(X_train, y_train)

test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          drop_last=False,
                          num_workers=workers)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False,
                         drop_last=False,
                         num_workers=workers)


# Feed-Forward Neural Network Model (N hidden layer)
class MLP(nn.Module):
    def __init__(self, layer_sizes: Iterable[int]):
        super(MLP, self).__init__()

        self.layers = []

        for i in range(1, len(layer_sizes)):
            if i < (len(layer_sizes) - 1):
                layer = nn.Sequential(
                    nn.BatchNorm1d(layer_sizes[i-1]),
                    nn.Linear(layer_sizes[i-1], layer_sizes[i]),
                    nn.ReLU())
            else: # last layer
                layer = nn.Sequential(
                    nn.BatchNorm1d(layer_sizes[i-1]),
                    nn.Linear(layer_sizes[i-1], layer_sizes[i]),
                    nn.Sigmoid())

            layer_name = "layer_{}".format(i)
            setattr(self, layer_name, layer)
            self.layers.append(layer_name)

        self._init_weigths()
        self.initial_weights = deepcopy(self.state_dict())
        return

    def _init_weigths(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.uniform_(-.15, .15)
                module.bias.data.zero_()

    def forward(self, x):
        for layer in self.layers:
            x = getattr(self, layer)(x)
        return x


def _train(model, dataloader, loss_func, optimizer):
    """
    Epoch train loop

    Iterate over minibatchs from dataloader doing forward + backprop
    """
    model.train()
    for i, (X, y) in enumerate(dataloader):
        y = y.float()

        # Convert tensor to Variable
        X = Variable(X)
        y = Variable(y)

        # zero the gradient buffer
        optimizer.zero_grad()

        # Forward
        y_pred = model(X).squeeze()

        weights = compute_sample_weight(y.data)
        weights = Variable(weights)

        loss = loss_func(y_pred, y, weights)

        # Backward + Optimize
        loss.backward()
        optimizer.step()


def _evaluate(model, dataloader, loss_func):
    """
    Calculate model metrics
    """
    loss = pt.FloatTensor()
    correct = pt.ByteTensor()

    model.eval()
    for i, (X, y) in enumerate(dataloader):
        y = y.float()

        X = Variable(X)
        y = Variable(y)

        y_pred = model(X).squeeze()

        weights = compute_sample_weight(y.data)
        weights = Variable(weights)

        batch_loss = loss_func(y_pred, y, weights)

        loss = pt.cat((loss, batch_loss.data))

        pred = y_pred.data > .5
        target = y.data > .5
        pred_correct = (pred == target)

        correct = pt.cat((correct, pred_correct))

    metrics = {
        "loss": loss.mean(),
        "acc": correct.float().mean()
    }
    return metrics


model = MLP(layer_sizes)
optimizer = pt.optim.Adam(model.parameters(), lr=0.001)
loss_func = binary_cross_entropy

for epoch in range(epochs):
    _train(model, train_loader, loss_func, optimizer)
    train_metrics = _evaluate(model, train_loader, loss_func)
    test_metrics = _evaluate(model, test_loader, loss_func)

    train_loss = train_metrics["loss"]
    test_loss = test_metrics["loss"]
    train_acc = train_metrics["acc"]
    test_acc = test_metrics["acc"]

    logger.scalar_summary("loss/train", train_loss, epoch)
    logger.scalar_summary("loss/test", test_loss, epoch)
    logger.scalar_summary("acc/train", train_acc, epoch)
    logger.scalar_summary("acc/test", test_acc, epoch)

    for param, values in model.named_parameters():
        if not param.startswith("layer"):
            continue

        tag = param.split('.')
        tag.pop(1)
        tag = "/".join(tag)
        logger.histo_summary(tag, values.data.numpy(), epoch, bins=1000)
        logger.histo_summary(tag+'/grad', values.grad.data.numpy(), epoch, bins=1000)

    print("{:2d} -> Train loss: {:.4f}\t Test loss: {:.4f}\t"
          "Train acc: {:.4f}\t Test acc: {:.4f}"
          .format(epoch, train_loss, test_loss, train_acc, test_acc))
