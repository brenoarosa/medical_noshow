"""
Script responsable to train a model
"""

from typing import Iterable
from copy import deepcopy

import numpy as np
import joblib

import torch as pt
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.functional import (binary_cross_entropy_with_logits, mse_loss,
                                 binary_cross_entropy)

from tensorboard import SummaryWriter
from utils.class_weights import compute_sample_weight

def _get_tensor_dataloader(tensors: Iterable[tuple], batch_size=1024,
                           shuffle=True, drop_last=False, n_cpu=None):
    """
    Transform torch tensors into dataloaders

    Args:
        tensors: iterable of tuples
            Iterable containing tuples of (X, y) torch tensors
        n_cpu: int (optional)
            Number of cpus to load from.
            Defaults to all.

    Returns:
        dataloaders: tuple
           tuple of DataLoader objects
    """

    if not n_cpu:
        n_cpu = joblib.cpu_count()

    dataloaders = []
    for (X, y) in tensors:
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            num_workers=n_cpu)
        dataloaders.append(loader)

    return tuple(dataloaders)


def _train(model, dataloader, loss_func, optimizer):
    """
    Epoch train loop

    Iterate over minibatchs from dataloader doing forward + backprop
    """
    model.train()
    for (X, y) in dataloader:
        y = y.float()

        # Convert tensor to Variable
        X = Variable(X)
        y = Variable(y)

        # zero the gradient buffer
        optimizer.zero_grad()

        # Forward
        y_pred = model(X).squeeze()
        weights = compute_sample_weight(y)
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

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    model.eval()
    for (X, y) in dataloader:
        y = y.float()

        X = Variable(X)
        y = Variable(y)

        y_pred = model(X).squeeze()
        weights = compute_sample_weight(y)
        batch_loss = loss_func(y_pred, y, weights)

        loss = pt.cat((loss, batch_loss.data))

        pred = y_pred.data > .5
        target = y.data > .5
        pred_correct = (pred == target)
        pred_wrong = (pred != target)

        correct = pt.cat((correct, pred_correct))

        target_positive = (y.data == 1)
        target_negative = (y.data == 0)

        batch_tp = pt.sum(pred_correct * target_positive)
        batch_tn = pt.sum(pred_correct * target_negative)
        batch_fp = pt.sum(pred_wrong * target_negative)
        batch_fn = pt.sum(pred_wrong * target_positive)

        tp += batch_tp
        tn += batch_tn
        fp += batch_fp
        fn += batch_fn

    metrics = {
        "loss": loss.mean(),
        "acc": correct.float().mean(),
        "f1": 2*tp / (2*tp + fn + fp)
    }
    return metrics


def train_model(model, tensors, optimizer, loss_func, **kwargs):

    batch_size = kwargs.pop("batch_size", 1024)
    epochs = kwargs.pop("epochs", 50)

    eps = kwargs.pop("eps", 1e-08)

    model_name = kwargs.pop("model_name", "model")

    early_stop = kwargs.pop("early_stop")

    log_base_path = kwargs.pop("log_base_path", "./log/")
    save_base_path = kwargs.pop("save_base_path", "./trained_models/")

    log_path = log_base_path + model_name
    save_path = save_base_path + model_name + ".p"
    logger = SummaryWriter(log_path)

    train_loader, val_loader = _get_tensor_dataloader(tensors, batch_size=batch_size)

    print("Training [{}]".format(model_name))
    print("| epoch | train loss | val loss | train acc | val acc | train F1 | val F1  | best model |")

    lower_val_loss = {"epoch": 0, "value": np.inf}
    best_model = deepcopy(model)

    if early_stop:
        es_metric_history = np.zeros(epochs)

    for epoch in range(1, epochs+1):
        _train(model, train_loader, loss_func, optimizer)
        train_metrics = _evaluate(model, train_loader, loss_func)
        val_metrics = _evaluate(model, val_loader, loss_func)

        train_loss = train_metrics["loss"]
        val_loss = val_metrics["loss"]
        train_acc = train_metrics["acc"]
        val_acc = val_metrics["acc"]
        train_f1 = train_metrics["f1"]
        val_f1 = val_metrics["f1"]

        # logging scalars
        logger.add_scalar("loss/train", train_loss, epoch)
        logger.add_scalar("loss/val", val_loss, epoch)
        logger.add_scalar("acc/train", train_acc, epoch)
        logger.add_scalar("acc/val", val_acc, epoch)
        logger.add_scalar("f1/train", train_f1, epoch)
        logger.add_scalar("f1/val", val_f1, epoch)

        # logging histogram
        for param, values in model.named_parameters():
            if not param.startswith("layer"):
                continue

            tag = param.split('.')
            tag.pop(1)
            tag = "/".join(tag)
            logger.add_histogram(tag, values.data.numpy(), epoch, bins="auto")
            logger.add_histogram(tag+'/grad', values.grad.data.numpy(), epoch, bins="auto")

        # Saves best model
        new_best_model = False
        if val_loss < lower_val_loss["value"]:
            lower_val_loss["value"] = val_loss
            lower_val_loss["epoch"] = epoch
            best_model = deepcopy(model)
            new_best_model = True

        print("| {: 5d} | {: 10.4f} | {: 8.4f} | {: 9.4f} | {: 7.4f} | {: 8.4f} | {: 6.4f} | {:10d} |"
              .format(epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, new_best_model))

        # Early stoping
        if early_stop:
            current_idx = (epoch-1)
            start_idx = np.maximum(0, current_idx - early_stop["patience"])

            es_metric = locals().get(early_stop["monitor"])
            es_metric_history[epoch-1] = es_metric

            if early_stop["mode"] == "min":
                stationary = es_metric >= (es_metric_history[start_idx:current_idx] - early_stop["min_delta"])
            if early_stop["mode"] == "max":
                stationary = es_metric <= (es_metric_history[start_idx:current_idx] + early_stop["min_delta"])

            if np.sum(stationary) == early_stop["patience"]:
                break

    pt.save(best_model, save_path)
    print("Saved [{}] from epoch {}".format(model_name, lower_val_loss["epoch"]))

    graph_pass = model(Variable(pt.FloatTensor(1, model.layer_1[1].in_features), requires_grad=True))
    logger.add_graph(model, graph_pass)

    return best_model
