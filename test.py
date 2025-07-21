import numpy as np


def tanh(x):
    return np.tanh(x)


def forward_linear(x, w, b):
    return np.dot(x, w.T) + b


def forward_policy(obs, weights):
    x = forward_linear(
        obs,
        weights["mlp_extractor.policy_net.0.weight"],
        weights["mlp_extractor.policy_net.0.bias"],
    )
    x = tanh(x)
    x = forward_linear(
        x,
        weights["mlp_extractor.policy_net.2.weight"],
        weights["mlp_extractor.policy_net.2.bias"],
    )
    x = tanh(x)
    action = forward_linear(x, weights["action_net.weight"], weights["action_net.bias"])
    return action


def normalize_lidar(lidar_array, max_value=1000.0):
    """
    lidar_array: 1D or 2D numpy array
    max_value: 정규화 기준 최대값
    return: 0~1 사이로 정규화된 배열
    """
    lidar_array = np.clip(lidar_array, 0, max_value)
    return lidar_array / max_value


import jchm
import numpy as np
import time

steering = 0
accerlation = 0

i = 0

while True:
    _, dist_array = jchm.lidar.get_lidar()
    front_dist = np.concatenate((dist_array[:35], dist_array[-35:]))
    lidar = normalize_lidar(front_dist)

    obs = np.concatenate([lidar, np.array([accerlation]), np.array([steering])])

    action = forward_policy(obs, weights)

    print(action)

    jchm.control.set_motor(action[1] * 10, action[1] * 10, 1)

    time.sleep(1)
