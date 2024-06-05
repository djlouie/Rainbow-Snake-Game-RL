from snake import Game
import random
import pygame
import math

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# if GPU is to be used
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def epsilon_greedy(obs):
    # epsilon decay policy
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
        math.exp(-1 * episodes_done / epsilon_decay)

    print(epsilon)

    # epsilon greedy policy
    sample = random.random()
    if sample > epsilon:
        print('exploitation')
        with torch.no_grad():
            return torch.argmax(policy_net(obs)).item()

    else:
        print('exploration')
        rand_action = random.randint(0,3)
        return rand_action
    

def observation(game):
    obs_matrix = torch.zeros(y_axis_size, x_axis_size)

    black = pygame.Color(0, 0, 0)
    white = pygame.Color(255, 255, 255)
    red = pygame.Color(255, 0, 0)
    orange = pygame.Color(255, 127, 0)
    yellow =pygame.Color(255, 255, 0)
    green = pygame.Color(0, 255, 0)
    blue = pygame.Color(0, 0, 255)
    indigo = pygame.Color(75, 0, 130)
    violet = pygame.Color(148, 0, 211)
    rainbow = [red, orange, yellow, green, blue, indigo, violet]

    for color, seg in zip(game.snake_colors, game.snake_body):
        x, y = seg
        x = int(x / 10)
        y = int(y / 10)

        if color == rainbow[0]:
            obs_matrix[y][x] = 50
        elif color == rainbow[1]:
            obs_matrix[y][x] = 51
        elif color == rainbow[2]:
            obs_matrix[y][x] = 52
        elif color == rainbow[3]:
            obs_matrix[y][x] = 53
        elif color == rainbow[4]:
            obs_matrix[y][x] = 54
        elif color == rainbow[5]:
            obs_matrix[y][x] = 55
        elif color == rainbow[6]:
            obs_matrix[y][x] = 56

    for ind, fruit in enumerate(game.fruit_position):
        x, y = fruit
        x = int(x / 10)
        y = int(y / 10)

        if ind == 0:
            obs_matrix[y][x] = 0
        elif ind == 1:
            obs_matrix[y][x] = 1
        elif ind == 2:
            obs_matrix[y][x] = 2
        elif ind == 3:
            obs_matrix[y][x] = 3
        elif ind == 4:
            obs_matrix[y][x] = 4
        elif ind == 5:
            obs_matrix[y][x] = 5
        elif ind == 6:
            obs_matrix[y][x] = 6
    
    return obs_matrix

def optimize_model(obs, next_obs, terminated, reward):

    # Compute Q(s_t, a)
    state_action_values = policy_net(torch.flatten(obs))

    # Compute V(s_{t+1})
    if terminated:
        next_state_values = torch.zeros(n_actions)
    else:
        with torch.no_grad():
            next_state_values = target_net(torch.flatten(next_obs))

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward

    # Compute Loss
    loss_function = nn.MSELoss()
    loss = loss_function(state_action_values, expected_state_action_values)

    # Optimize model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# hyperparameters
batch_size = None
gamma = 0.99
epsilon = 0.90
epsilon_start = 0.90
epsilon_end = 0.05
epsilon_decay = 100
tau = 0.005
lr = 1e-3

# action space and observation space size
x_window_size = 720
y_window_size = 480
x_axis_size = int(x_window_size / 10)
y_axis_size = int(y_window_size / 10)
n_actions = 4  # up down left right
n_observations = int(x_axis_size * y_axis_size)

# initialize neural networks
policy_net = DQN(n_observations, n_actions)  #.to(device)
target_net = DQN(n_observations, n_actions)  #.to(device)

# set target net to have the same weights as policy net
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)

# create a steps counter
steps_done = 0
episodes_done = 0
n_episodes = 500

for ep in range(n_episodes):
    game = Game(x_window_size, y_window_size, episodes_done)
    obs = observation(game)

    while True:

        # select action
        action = epsilon_greedy(torch.flatten(obs))

        # save old score
        old_score = game.score

        print(action)

        # game step
        terminated = game.step(action)

        # get new reward
        if old_score < game.score:
            reward = 10
        elif terminated:
            reward = -100
        else:
            reward = 0.1

        # get new observation
        if terminated:
            new_obs = None
        else:
            new_obs = observation(game)

        # step of optimization
        optimize_model(obs, new_obs, terminated, reward)
        
        # update obs
        obs = new_obs

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
        target_net.load_state_dict(target_net_state_dict)

        if terminated:
            episodes_done += 1
            break

torch.save(policy_net, 'policy.pth')