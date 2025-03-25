#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 2025-03-22 17:50:45 Saturday

@author: Nikhil Kapila
"""

import torch,torchvision
import mlflow, os, sys 
from matplotlib import pyplot as plt
from etl import load_data
from typing import Type

class Pipeline:
    def __init__(
        self,
        exp_name: str,
        run_name: str,
        model: torch.nn.Module,
        optimizer: Type[torch.optim.Optimizer],
        lr: float = 0.001, #1e-3 is the default
        weight_decay: float = 0, # no reg is the default
        epochs:int = 10,
        device: str = 'cpu', 
        
        transforms = None, # for vision tasks, transforms,
        log_model = True, # Log model
        **args
        ):

        self.exp_name = exp_name
        self.run_name = run_name
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = self.optimizer = optimizer(self.model.parameters(), 
                                                    lr=self.lr, 
                                                    weight_decay=self.weight_decay)
        self.epochs = epochs
        self.device = device
        self.transforms = transforms
        self.log_model = log_model
        self.args = args
        self.params = {key: value for key, value in self.__dict__.items() if not key.startswith('__')}

    def load_data(self):
        pass
        

    def train(self):
        pass

    def eval(self):
        pass

    def plot(self):
        pass

    def start_pipeline(self):
        mlflow.set_experiment(experiment_name=self.exp_name)

        with mlflow.start_run() as run:
            print(f"\nExperiment ID: {run.info.experiment_id}")
            print(f"\nRun ID: {run.info.run_id}")
            print(f'Model is saved at: {os.path.join(os.getcwd()), os.path.join(run.info.experiment_id, run.info.run_id)}')
            print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
            print(f"Using device: {self.device}")

            # parameter logging
            mlflow.log_params(self.params)

            mlflow.set_tag(self.tag)

            

        

        
        
        
    