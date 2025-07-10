# */* coding = utf-8 */*
"""
modeling.py a function where i consolidate all my set of model that i want to test

@author = Faouzi Zakaria
Date: 10/07/2025
    
"""

import torch
import torch.nn as nn
from torch.optim import optim

# model
from sklearn.linear_model import LogisticRegression


class BinaryMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(BinaryMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x) # logit sans sigmoid ici
    
class Trainer:

    def __init__(self, model, device = None, lr = 0.01, epochs = 25, batch_size= 32):
        self.model = model
        self.device = device or ('cude' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.lr)
        self.model.to(self.device)

    
