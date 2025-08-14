import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class Network(nn.Module):
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action

        self.fc_other = nn.Linear(input_size - 90, 32)

        self.fc1 = nn.Linear(122, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, nb_action)

    def forward(self, state):
        lidar_data = state[:, :-2]
        other_data = state[:, -2:]

        x1 = lidar_data

        x2 = F.relu(self.fc_other(other_data))  # [1, 32]

        x = torch.cat([x1, x2], dim=1)  # [1, 64]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        batch_state, batch_next_state, batch_action, batch_reward = samples

        # Properly handle dtypes and return in correct order for learn()
        return (
            torch.cat(batch_state, 0),  # batch_state
            torch.cat(batch_reward, 0),  # batch_reward
            torch.cat(batch_action, 0),  # batch_action (already LongTensor)
            torch.cat(batch_next_state, 0),  # batch_next_state
        )

    def __len__(self):
        return len(self.memory)


class DQN(object):
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(capacity=100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.00005)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        self.speed_values = [0, 5, 10]
        self.steering_values = [-10, -5, 0, 5, 10]
        self.action_space = [
            (sp, st) for sp in self.speed_values for st in self.steering_values
        ]

    def select_action(self, state):
        with torch.no_grad():
            # Pass the state through the model to get Q-values.
            # The model and state should be on the same device (e.g., CPU or GPU).
            q_values = self.model(state)

            # Convert the Q-values into a probability distribution using the softmax function.
            # This ensures the values are non-negative and sum to 1, as required by multinomial.
            # This method of selecting an action is known as Boltzmann exploration.
            probs = F.softmax(q_values, dim=1)

            # Sample one action from the probability distribution.
            action = probs.multinomial(num_samples=1)

        # Return the selected action as a Python number.
        return action.item()

    def learn(self, batch_state, batch_reward, batch_action, batch_next_state):
        batch_outputs = (
            self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        )
        batch_next_outputs = self.model(batch_next_state).detach().max(1)[0]
        batch_targets = batch_reward + self.gamma * batch_next_outputs
        td_loss = F.smooth_l1_loss(batch_outputs, batch_targets)
        print(f"td_loss: {td_loss}")
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

    def update(self, new_state, new_reward):
        new_state = torch.Tensor(new_state).float()  # Add unsqueeze(0)
        self.memory.push(
            (
                self.last_state,
                new_state,
                torch.LongTensor([int(self.last_action)]),
                torch.Tensor([self.last_reward]),
            )
        )
        action = self.select_action(new_state)
        if len(self.memory) > 100:
            batch_state, batch_reward, batch_action, batch_next_state = (
                self.memory.sample(100)
            )
            self.learn(batch_state, batch_reward, batch_action, batch_next_state)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = new_reward

    def save(self):
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            "dqn/last_brain.pth",
        )

    def load(self):
        if os.path.isfile("dqn/last_brain.pth"):
            print("=> loading checkpoint...")
            checkpoint = torch.load("dqn/last_brain.pth")
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            print("Done!")
        else:
            print("No checkpoint found...")
