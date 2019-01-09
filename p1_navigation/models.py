import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC

HIDDEN_SIZES=[20, 20]


class DuelingQNetwork(nn.Module):
    "Dueling Q-Network Actor (Policy) Architecture"

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # divide into two parts:
        # the first part is the feature constructing part
        # the second part is the V+A double-head
        self.fc1 = nn.Linear(state_size, HIDDEN_SIZES[0])
        self.fc2 = nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1])
        self.output_v = nn.Linear(HIDDEN_SIZES[1], action_size)
        self.output_a = nn.Linear(HIDDEN_SIZES[1], action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        V = self.output_v(x)
        A = self.output_a(x)
        return V + A - A.mean(dim=1).unsqueeze(1)


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # model construction
        from collections import OrderedDict
        self.model = nn.Sequential(OrderedDict([
            ('fc1,', nn.Linear(state_size, HIDDEN_SIZES[0])),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(HIDDEN_SIZES[0],HIDDEN_SIZES[1])),
            ('relu2', nn.ReLU()),
            ('output', nn.Linear(HIDDEN_SIZES[1], action_size))
        ]))

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.model(state)