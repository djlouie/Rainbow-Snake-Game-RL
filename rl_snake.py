from snake import Game
import random

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(n_observations, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# hyperparameters
batch_size = None
gamma = 0.99
epsilon_start = 0.95
epsilon_end = 0.05
epsilon_decay = 50
tau = 0.005
lr = 1e-3

# action space and observation space size
n_actions = 4
n_observations = None

# initialize neural networks
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)

# set target net to have the same weights as policy net
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)

# create a steps counter
n_steps = 0


def epsilon_greedy(n_steps):
    ...

game = Game(100, 100)
while True:
    game.step()