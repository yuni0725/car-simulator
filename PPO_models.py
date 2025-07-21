import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPOModel(nn.Module):
    def __init__(self, input_size, nb_action):
        super(PPOModel, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action

        self.conv1 = nn.Conv1d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=7)
        self.conv3 = nn.Conv1d(32, 16, kernel_size=9)
        self.conv4 = nn.Conv1d(16, 1, kernel_size=41)

        self.fc_other = nn.Linear(input_size - 90, 32)

        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)

        self.policy_head = nn.Linear(128, nb_action)
        self.value_head = nn.Linear(128, 1)

    def forward(self, state):
        lidar_data = state[:, :-2]
        other_data = state[:, -2:]

        x1 = lidar_data.unsqueeze(1)
        x1 = F.relu(self.conv1(x1))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = F.relu(self.conv4(x1))
        x1 = x1.squeeze(1)

        x2 = F.relu(self.fc_other(other_data))

        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)

        return logits, value

    def get_action(self, state):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def evaluate(self, state, action):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return log_prob, entropy, value


class PPOAgent:
    def __init__(self, input_size, nb_action, lr=3e-4):
        self.model = PPOModel(input_size, nb_action)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_action(self, state):
        return self.model.get_action(state)

    def evaluate(self, state, action):
        return self.model.evaluate(state, action)

    def save(self, path="ppo/last_model.pth"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )
        print(f"[✓] PPO 모델 저장됨: {path}")

    def load(self, path="ppo/last_model.pth"):
        if os.path.isfile(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"[✓] PPO 모델 로드 완료: {path}")
        else:
            print(f"[x] 저장된 PPO 모델이 없음: {path}")
