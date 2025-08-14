import torch
import numpy as np
import torch.nn.functional as F
import csv
import os


def init_log(file_path="ppo_logs.csv"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["epoch", "reward_total", "policy_loss", "value_loss", "total_loss"]
        )


def log_epoch(file_path, epoch, reward_total, policy_loss, value_loss, total_loss):
    with open(file_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, reward_total, policy_loss, value_loss, total_loss])


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]  # 마지막 value=0으로 padding
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (1 - dones[t]) * values[t + 1] - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    return advantages


def process_obs(obs_dict):
    """Dict obs를 flat vector로 변환 (lidar + velocity + steering → 72차원)"""
    lidar = obs_dict["lidar_data"]
    vel = obs_dict["velocity"]
    steer = obs_dict["steering"]
    return np.concatenate([lidar, vel, steer], axis=-1)  # shape: (72,)


def train_ppo(
    agent,
    env,
    epochs=1000,
    rollout_steps=2048,
    batch_size=64,
    ppo_epochs=4,
    clip_eps=0.2,
    log_file="ppo_logs.csv",
    render=False,
):

    init_log(log_file)

    for epoch in range(epochs):
        # Storage
        obs_buffer, action_buffer, reward_buffer = [], [], []
        logprob_buffer, value_buffer, done_buffer = [], [], []

        obs = env.reset()
        total_reward = 0

        for _ in range(rollout_steps):
            obs_vector = process_obs(obs)
            obs_tensor = torch.tensor(obs_vector, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, value = agent.get_action(obs_tensor)

            action_index = action.item()
            next_obs, reward, done, _ = env.step(action_index)

            obs_buffer.append(obs_vector)
            action_buffer.append(action_index)
            reward_buffer.append(reward)
            logprob_buffer.append(log_prob.item())
            value_buffer.append(value.item())
            done_buffer.append(done)

            obs = next_obs
            total_reward += reward

            if render:
                env.render()

            if done:
                obs = env.reset()

        # Compute advantage and returns
        advantages = compute_gae(reward_buffer, value_buffer, done_buffer)
        returns = [adv + val for adv, val in zip(advantages, value_buffer)]

        # Convert to tensors
        obs_tensor = torch.tensor(obs_buffer, dtype=torch.float32)
        action_tensor = torch.tensor(action_buffer, dtype=torch.long)
        old_logprob_tensor = torch.tensor(logprob_buffer, dtype=torch.float32)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        adv_tensor = torch.tensor(advantages, dtype=torch.float32)
        adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        # PPO update
        last_policy_loss = last_value_loss = last_total_loss = 0.0

        for _ in range(ppo_epochs):
            indices = np.arange(rollout_steps)
            np.random.shuffle(indices)
            for i in range(0, rollout_steps, batch_size):
                batch_idx = indices[i : i + batch_size]

                b_obs = obs_tensor[batch_idx]
                b_actions = action_tensor[batch_idx]
                b_old_logprob = old_logprob_tensor[batch_idx]
                b_returns = returns_tensor[batch_idx]
                b_advantages = adv_tensor[batch_idx]

                new_logprob, entropy, value = agent.evaluate(b_obs, b_actions)
                ratio = (new_logprob - b_old_logprob).exp()

                clip_adv = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * b_advantages
                policy_loss = -torch.min(ratio * b_advantages, clip_adv).mean()
                value_loss = F.mse_loss(value, b_returns)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

                agent.optimizer.zero_grad()
                loss.backward()
                agent.optimizer.step()

                last_policy_loss = policy_loss.item()
                last_value_loss = value_loss.item()
                last_total_loss = loss.item()

        print(
            f"[Epoch {epoch + 1}] Reward: {total_reward:.2f} | Loss: {last_total_loss:.4f}"
        )
        log_epoch(
            log_file,
            epoch + 1,
            total_reward,
            last_policy_loss,
            last_value_loss,
            last_total_loss,
        )
        if (epoch + 1) % 10 == 0:
            agent.save(f"ppo/model_epoch_{epoch+1}.pth")
            print(f"[💾] 모델 저장됨: ppo/model_epoch_{epoch+1}.pth")


from environment import CarSimulations
from PPO_models import PPOAgent

env = CarSimulations()
agent = PPOAgent(input_size=92, nb_action=15)
agent.load("ppo/3car_900.pth")

train_ppo(
    agent,
    env,
    epochs=100,
    rollout_steps=2048,
    batch_size=64,
    ppo_epochs=4,
    clip_eps=0.2,
    log_file="ppo/ppo_logs.csv",
    render=True,
)
