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

import torch
import numpy as np

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

        if learner.architecture.external_model_path is None:
            raise Exception("embedded architecture definition not supported"
                " for PyTorch models")
        elif learner.architecture.external_model_path is not None and \
             learner.architecture.external_model_format is not None:
            if learner.architecture.external_model_format == "pytorch-module":
                if os.path.isfile(learner.architecture.external_model_path):
                    if learner.architecture.external_model_class_name is None:
                        raise Exception("PyTorch model class name not specified")
                    else:
                        module_name = learner.architecture.external_model_path
                        module_name = module_name.replace(".py", "")
                        module_name = module_name.replace("/", ".")
                        module_name = module_name.replace("\\", ".")
                        
                        module = importlib.import_module(module_name)
                        torch_model_class = getattr(module, learner.architecture.external_model_class_name)
                        learner.model = torch_model_class()
                else:
                    raise Exception("PyTorch model class file does not exist: " + 
                                    learner.architecture.external_model_path)
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
            learner.criterion = TorchHelper.get_loss(learner.loss_hyperparameters)
            
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
        
        batch_size = int(learner.training_process_hyperparameters["batch_size"])

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
        num_epochs: int = int(learner.training_process_hyperparameters["epochs"])

        #best_model_wts = copy.deepcopy(learner.model.state_dict())
        #best_acc: float = 0.0

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
                        outputs = learner.model(x.float())
                        loss = learner.criterion(outputs,
                                                 torch.argmax(y.float(), dim=1))

                        # backward + optimize only if in training phase
                        if phase == "training":
                            loss.backward()
                            learner.optimizer.step()

                        # statistics
                        y_hat = torch.argmax(outputs, dim=1)
                        running_loss += loss.item() * x.size(0)
                        running_correct += torch.sum(y_hat == torch.argmax(y.float(), dim=1))

                epoch_loss = running_loss / len(data_loader.dataset)
                epoch_acc = running_correct.float() / len(data_loader.dataset)

                print("{} loss: {:.4f} accuracy: {:.4f}".format(
                      phase, epoch_loss, epoch_acc))

                # deep copy the model
                #if phase == "validation" and epoch_acc > best_acc:
                #    best_acc = epoch_acc
                #    best_model_wts = copy.deepcopy(model.state_dict())

    @staticmethod
    def save_model(learner: Learner, model_name: str = "") -> None:
        if model_name != "":
            os.makedirs(learner.output_dir + model_name)

        torch.save(learner.model.state_dict(), learner.output_dir + \
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
        learner.model.load_state_dict(torch.load(learner.output_dir + \
            model_name + "/saved_model.pth"))

    @staticmethod
    def predict(learner: Learner, dataset: torch.utils.data.Dataset,
                final_activation_function = None):
        """ This is the forward calculation from x to y
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """
        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(learner.training_process_hyperparameters["batch_size"]),
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

                raw_logits = learner.model(x.float())
                if final_activation_function is None:
                    preds = preds + raw_logits.tolist()
                elif final_activation_function == "softmax":
                    preds = preds + \
                        torch.nn.functional.softmax(raw_logits, dim=1).tolist()
                elif final_activation_function == "sigmoid":
                    preds = preds + \
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
            batch_size=int(learner.training_process_hyperparameters["batch_size"]),
            shuffle=False)

        # GPU or CPU?
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        learner.model = learner.model.to(device)

        learner.model.eval()
        with torch.no_grad():
            for x, y in data_loader:
                # transfer to device
                x = x.to(device)
                y = y.to(device)

                outputs = learner.model(x.float())
                y_hat = torch.argmax(outputs, dim=1)
                loss = learner.criterion(outputs,
                                       torch.argmax(y.float(), dim=1))

        return [0.0, 0.0]

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
            else:
                raise Exception("unknown loss specified: " + loss)
        else:
            raise Exception("no loss specified")
