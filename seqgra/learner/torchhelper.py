"""MIT - CSAIL - Gifford Lab - seqgra

PyTorch learner helper class

@author: Konstantin Krismer
"""
from ast import literal_eval
from typing import List, Any
import os
import sys
import random
import pkg_resources
import importlib
import logging

import torch
import numpy as np
from ignite.engine import Events
from ignite.engine import create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping, ModelCheckpoint

from seqgra.learner.learner import Learner


class TorchHelper:
    @staticmethod
    def to_bool(x: str) -> bool:
        x = x.strip()
        if x == "True":
            return True
        elif x == "False":
            return False
        else:
            raise Exception("'" + str(x) +
                            "' must be either 'True' or 'False'")

    @staticmethod
    def create_model(learner: Learner) -> None:
        learner.set_seed()
        path = learner.architecture.external_model_path
        class_name = learner.architecture.external_model_class_name

        if path is None:
            raise Exception("embedded architecture definition not supported"
                            " for PyTorch models")
        elif path is not None and \
                learner.architecture.external_model_format is not None:
            if learner.architecture.external_model_format == "pytorch-module":
                if os.path.isfile(path):
                    if class_name is None:
                        raise Exception(
                            "PyTorch model class name not specified")
                    else:
                        location = path
                        module_spec = importlib.util.spec_from_file_location(
                            "model", location)
                        torch_model_module = importlib.util.module_from_spec(
                            module_spec)
                        module_spec.loader.exec_module(torch_model_module)
                        torch_model_class = getattr(torch_model_module,
                                                    class_name)
                        learner.model = torch_model_class()
                else:
                    raise Exception("PyTorch model class file does not exist: " +
                                    path)
            else:
                raise Exception("unsupported PyTorch model format: " +
                                learner.architecture.external_model_format)
        else:
            raise Exception("neither internal nor external architecture "
                            "definition provided")

        if learner.optimizer_hyperparameters is None:
            raise Exception("optimizer undefined")
        else:
            learner.optimizer = TorchHelper.get_optimizer(
                learner.optimizer_hyperparameters,
                learner.model.parameters())

        if learner.loss_hyperparameters is None:
            raise Exception("loss undefined")
        else:
            learner.criterion = TorchHelper.get_loss(
                learner.loss_hyperparameters)

        if learner.metrics is None:
            raise Exception("metrics undefined")

    @staticmethod
    def print_model_summary(learner: Learner):
        print(learner.model)

    @staticmethod
    def set_seed(learner: Learner) -> None:
        random.seed(learner.seed)
        np.random.seed(learner.seed)
        torch.manual_seed(learner.seed)

    @staticmethod
    def train_model(learner: Learner,
                    training_dataset: torch.utils.data.Dataset,
                    validation_dataset: torch.utils.data.Dataset) -> None:
        if learner.model is None:
            learner.create_model()

        batch_size = int(
            learner.training_process_hyperparameters["batch_size"])

        # init data loaders
        training_loader = torch.utils.data.DataLoader(
            training_dataset,
            batch_size=batch_size,
            shuffle=bool(learner.training_process_hyperparameters["shuffle"]))

        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False)

        # GPU or CPU?
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        learner.model = learner.model.to(device)

        # training loop
        trainer = create_supervised_trainer(learner.model, learner.optimizer,
                                            learner.criterion, device=device)
        train_evaluator = create_supervised_evaluator(
            learner.model,
            metrics=TorchHelper.get_metrics(learner),
            device=device)
        val_evaluator = create_supervised_evaluator(
            learner.model,
            metrics=TorchHelper.get_metrics(learner),
            device=device)

        logging.getLogger("ignite.engine.engine.Engine").setLevel(
            logging.WARNING)

        num_epochs: int = int(
            learner.training_process_hyperparameters["epochs"])

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            print("epoch {}/{}".format(trainer.state.epoch, num_epochs))
            train_evaluator.run(training_loader)
            metrics = train_evaluator.state.metrics
            print(TorchHelper._format_metrics_output(metrics, "training set"))

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(trainer):
            val_evaluator.run(validation_loader)
            metrics = val_evaluator.state.metrics
            print(TorchHelper._format_metrics_output(metrics,
                                                     "validation set"))

        # save best model
        def score_fn(engine):
            if "loss" in learner.metrics:
                score = engine.state.metrics["loss"]
                score = -score
            elif "accuracy" in learner.metrics:
                score = engine.state.metrics["accuracy"]
            else:
                raise Exception("no metric to track performance")
            return score

        best_model_saver_handler = ModelCheckpoint(
            learner.output_dir + "training/models",
            score_function=score_fn,
            filename_prefix="best",
            n_saved=1,
            create_dir=True)
        val_evaluator.add_event_handler(Events.COMPLETED,
                                        best_model_saver_handler,
                                        {"model": learner.model})

        # early stopping callback
        if bool(learner.training_process_hyperparameters["early_stopping"]):
            es_handler = EarlyStopping(patience=2,
                                       score_function=score_fn,
                                       trainer=trainer,
                                       min_delta=0)
            val_evaluator.add_event_handler(Events.COMPLETED, es_handler)

        trainer.run(training_loader, max_epochs=num_epochs)

    @staticmethod
    def _format_metrics_output(metrics, set_label):
        message: List[str] = [set_label + " metrics:\n"]
        message += [" - " + metric + ": " + str(metrics[metric]) + "\n"
                    for metric in metrics]
        return "".join(message).rstrip()

    @staticmethod
    def train_model_basic(learner: Learner,
                          training_dataset: torch.utils.data.Dataset,
                          validation_dataset: torch.utils.data.Dataset) -> None:
        if learner.model is None:
            learner.create_model()

        batch_size = int(
            learner.training_process_hyperparameters["batch_size"])

        # init data loaders
        training_loader = torch.utils.data.DataLoader(
            training_dataset,
            batch_size=batch_size,
            shuffle=bool(learner.training_process_hyperparameters["shuffle"]))

        validation_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False)

        # GPU or CPU?
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        learner.model = learner.model.to(device)

        # training loop
        num_epochs: int = int(
            learner.training_process_hyperparameters["epochs"])

        for epoch in range(num_epochs):
            print("epoch {}/{}".format(epoch + 1, num_epochs))

            for phase in ["training", "validation"]:
                if phase == "training":
                    learner.model.train()
                    data_loader = training_loader
                else:
                    learner.model.eval()
                    data_loader = validation_loader

                running_loss: float = 0.0
                running_correct: int = 0

                for x, y in data_loader:
                    # transfer to device
                    x = x.to(device)
                    y = y.to(device)

                    # zero the parameter gradients
                    learner.optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == "training"):
                        outputs = learner.model(x)
                        loss = learner.criterion(outputs, y)

                        # backward + optimize only if in training phase
                        if phase == "training":
                            loss.backward()
                            learner.optimizer.step()

                        # statistics
                        y_hat = torch.argmax(outputs, dim=1)
                        running_loss += loss.item() * x.size(0)
                        running_correct += torch.sum(y_hat == y)

                epoch_loss = running_loss / len(data_loader.dataset)
                epoch_acc = running_correct.float() / len(data_loader.dataset)

                print("{} - loss: {:.3f}, accuracy: {:.3f}".format(
                      phase, epoch_loss, epoch_acc))

    @staticmethod
    def save_model(learner: Learner, model_name: str = "") -> None:
        if model_name != "":
            os.makedirs(learner.output_dir + model_name)

        torch.save(learner.model.state_dict(), learner.output_dir +
                   model_name + "/saved_model.pth")

        # save session info
        learner.write_session_info()

    @staticmethod
    def write_session_info(learner: Learner) -> None:
        with open(learner.output_dir + "session-info.txt", "w") as session_file:
            session_file.write("seqgra package version: " +
                               pkg_resources.require("seqgra")[0].version + "\n")
            session_file.write("PyTorch version: " + torch.__version__ + "\n")
            session_file.write("NumPy version: " + np.version.version + "\n")
            session_file.write("Python version: " + sys.version + "\n")

    @staticmethod
    def load_model(learner: Learner, model_name: str = "") -> None:
        TorchHelper.create_model(learner)
        learner.model.load_state_dict(torch.load(learner.output_dir +
                                                 model_name + "/saved_model.pth"))

    @staticmethod
    def predict(learner: Learner, dataset: torch.utils.data.Dataset,
                final_activation_function=None):
        """ This is the forward calculation from x to y
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(
                learner.training_process_hyperparameters["batch_size"]),
            shuffle=False)

        # GPU or CPU?
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        learner.model = learner.model.to(device)

        preds = []
        learner.model.eval()
        with torch.no_grad():
            for x in data_loader:
                # transfer to device
                x = x.to(device)

                raw_logits = learner.model(x)
                if final_activation_function is None:
                    preds += raw_logits.tolist()
                elif final_activation_function == "softmax":
                    preds += \
                        torch.nn.functional.softmax(raw_logits, dim=1).tolist()
                elif final_activation_function == "sigmoid":
                    preds += \
                        torch.nn.functional.sigmoid(raw_logits, dim=1).tolist()

        return np.array(preds)

    @staticmethod
    def get_num_params(learner: Learner):
        if learner.model is None:
            learner.create_model()
        return len(learner.model.parameters())

    @staticmethod
    def evaluate_model(learner: Learner, dataset: torch.utils.data.Dataset):
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(
                learner.training_process_hyperparameters["batch_size"]),
            shuffle=False)

        # GPU or CPU?
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        learner.model = learner.model.to(device)

        running_loss: float = 0.0
        running_correct: int = 0

        learner.model.eval()
        with torch.no_grad():
            for x, y in data_loader:
                # transfer to device
                x = x.to(device)
                y = y.to(device)

                outputs = learner.model(x)
                loss = learner.criterion(outputs, y)

                # statistics
                y_hat = torch.argmax(outputs, dim=1)
                running_loss += loss.item() * x.size(0)
                running_correct += torch.sum(y_hat == y)

        overall_loss = running_loss / len(data_loader.dataset)
        overall_accuracy = running_correct.float().item() / len(data_loader.dataset)

        return {"loss": overall_loss, "accuracy": overall_accuracy}

    @staticmethod
    def get_optimizer(optimizer_hyperparameters, model_parameters):
        if "optimizer" in optimizer_hyperparameters:
            optimizer = \
                optimizer_hyperparameters["optimizer"].lower().strip()

            if "learning_rate" in optimizer_hyperparameters:
                learning_rate = float(
                    optimizer_hyperparameters["learning_rate"].strip())
            else:
                learning_rate = 0.001

            if optimizer == "sgd":
                if "momentum" in optimizer_hyperparameters:
                    momentum = float(
                        optimizer_hyperparameters["momentum"].strip())
                else:
                    momentum = 0.0

                return torch.optim.SGD(
                    model_parameters,
                    lr=learning_rate,
                    momentum=momentum)
            else:
                raise Exception("unknown optimizer specified: " + optimizer)
        else:
            raise Exception("no optimizer specified")

    @staticmethod
    def get_loss(loss_hyperparameters):
        if "loss" in loss_hyperparameters:
            loss = loss_hyperparameters["loss"].lower().replace(
                "_", "").strip()
            if loss == "crossentropyloss":
                return torch.nn.CrossEntropyLoss()
            elif loss == "nllloss":
                return torch.nn.NLLLoss()
            elif loss == "bcewithlogitsloss":
                return torch.nn.BCEWithLogitsLoss()
            else:
                raise Exception("unknown loss specified: " + loss)
        else:
            raise Exception("no loss specified")

    @staticmethod
    def get_metrics(learner: Learner):
        is_multilabel = learner.learner_type == "multi-label classification"
        metrics_dict = dict()
        for metric in learner.metrics:
            metric = metric.lower().strip()
            if metric == "loss":
                metrics_dict[metric] = Loss(learner.criterion)
            elif metric == "accuracy":
                metrics_dict[metric] = Accuracy(is_multilabel=is_multilabel)
            else:
                logging.warn("unknown metric: " + metric)
        return metrics_dict
