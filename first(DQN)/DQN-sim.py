import numpy as np
from simulations.world import World
from simulations.agents import (
    Car,
    RectangleBuilding,
    Pedestrian,
    Painting,
    RectangleEntity,
    CircleEntity,
)
from simulations.geometry import Point
import time
from fast_lidar import calculate_lidar_distances
import sys
import torch
from random import uniform

dt = 0.1  # time steps in terms of seconds. In other words, 1/dt is the FPS.
w = World(
    dt, width=120, height=70, ppm=6
)  # The world is 120 meters by 120 meters. ppm is the pixels per meter.

# Let's add some sidewalks and RectangleBuildings.
# A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks.
# A RectangleBuilding object is also static -- it does not move. But as opposed to Painting, it can be collided with.
# For both of these objects, we give the center point and the size.

w.add(RectangleBuilding(Point(20, 60), Point(50, 120)))
w.add(RectangleBuilding(Point(100, 60), Point(50, 120)))

for i in range(11):
    w.add(Painting(Point(60, 12 * i), Point(0.5, 2), "white"))

# A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
c1 = Car(Point(60, 0), np.pi / 2)
c1.velocity = Point(0, 5)
w.add(c1)

c2 = Car(Point(60, 35), np.pi / 2, "blue")
w.add(c2)

w.render()


def lidar_data(car, world, num_rays=90, max_distance=50.0, fov_deg=180.0):
    rect_obstacles = []
    circle_obstacles = []

    for agent in world.agents:
        if not agent.collidable or agent is car:
            continue

        if isinstance(agent, RectangleEntity):
            rect_obstacles.append(
                [
                    agent.center.x,
                    agent.center.y,
                    agent.size.x,
                    agent.size.y,
                    agent.heading,
                ]
            )
        elif isinstance(agent, CircleEntity):
            circle_obstacles.append([agent.center.x, agent.center.y, agent.radius])

    rects_np = (
        np.array(rect_obstacles, dtype=np.float32)
        if rect_obstacles
        else np.empty((0, 5), dtype=np.float32)
    )
    circles_np = (
        np.array(circle_obstacles, dtype=np.float32)
        if circle_obstacles
        else np.empty((0, 3), dtype=np.float32)
    )

    return calculate_lidar_distances(
        car.center.x,
        car.center.y,
        car.heading,
        rects_np,
        circles_np,
        num_rays=num_rays,
        max_distance=max_distance,
        fov_deg=fov_deg,
        num_steps=50,
    )


def normalize_lidar(lidar_array, max_value=50.0):
    """
    lidar_array: 1D or 2D numpy array
    max_value: 정규화 기준 최대값
    return: 0~1 사이로 정규화된 배열
    """
    lidar_array = np.clip(lidar_array, 0, max_value)
    return lidar_array / max_value


goal = Point(60, uniform(50, 69))

from deep_Q_learning import DQN

brain = DQN(input_size=92, nb_action=15, gamma=0.9)

brain.load()

episode_steps = 0
max_steps = 300  # Optional timeout
episode_reward = 0.0
episode_count = 0

prev_steering_cmd = 0.0


def speed_to_acceleration(current_speed, target_speed, dt=0.1, friction=0):
    """Convert target speed to required acceleration"""
    # From entities.py: new_speed = current_speed + (inputAcceleration - friction) * dt
    # Solving for inputAcceleration:
    required_acceleration = (target_speed - current_speed) / dt + friction
    return required_acceleration


while True:
    prev_position = w.dynamic_agents[0].center

    w.tick()  # This ticks the world for one time step (dt second)
    w.render()

    speed = w.dynamic_agents[0].velocity.y
    steering = w.dynamic_agents[0].inputSteering

    normalized_lidar = normalize_lidar(lidar_data(w.dynamic_agents[0], w))

    lidar_tensor = torch.tensor(normalized_lidar, dtype=torch.float32).flatten()
    other_data = torch.tensor([speed, steering], dtype=torch.float32)
    state = torch.cat([lidar_tensor, other_data], dim=0).unsqueeze(0)

    action_index = brain.select_action(state)
    target_speed, steering_cmd = brain.action_space[action_index]

    # Convert speed command to acceleration
    current_speed = w.dynamic_agents[0].velocity.y
    acceleration_cmd = speed_to_acceleration(current_speed, target_speed, dt)

    # Use only set_control with calculated acceleration
    w.dynamic_agents[0].set_control(steering_cmd, acceleration_cmd)

    # 보상함수 계산

    done = False

    distance = np.linalg.norm(
        [w.dynamic_agents[0].center.x - goal.x, w.dynamic_agents[0].center.y - goal.y]
    )
    prev_distance = np.linalg.norm([prev_position.x - goal.x, prev_position.y - goal.y])
    reward = 5.0 * (prev_distance - distance) / max(distance, 1.0)

    steering_change_penalty = -abs(steering_cmd - prev_steering_cmd) / 10.0
    prev_steering_cmd = steering_cmd

    reward += target_speed * 0.05
    reward += steering_change_penalty * 0.05

    if (
        w.collision_exists()
        or w.dynamic_agents[0].center.y < 0
        or w.dynamic_agents[0].center.y > 70
    ):
        print("Collision detected!")
        reward -= 2.0
        done = True

    elif (
        np.linalg.norm(
            [
                w.dynamic_agents[0].center.x - goal.x,
                w.dynamic_agents[0].center.y - goal.y,
            ]
        )
        < 5.0
    ):
        reward += 3.0
        done = True

    elif episode_steps > max_steps:
        reward -= 0.5
        done = True

    # 학습
    brain.update(state, reward)
    episode_reward += reward
    episode_steps += 1

    # 에피소드 종료 처리
    if done:
        episode_count += 1
        print(f"Last reward: {reward:.2f}, Steps: {episode_steps}")

        if episode_count % 20 == 0:
            brain.save()

        with open("dqn/training_log.csv", "a") as f:
            f.write(f"{episode_count},{episode_reward:.2f},{episode_steps}\n")

        # 에이전트 초기화
        w.dynamic_agents[0].center = Point(60, uniform(0, 20))
        w.dynamic_agents[0].velocity = Point(0, uniform(0, 5))
        w.dynamic_agents[0].inputSteering = 0.0

        # 상태 초기화
        episode_reward = 0.0
        episode_steps = 0

        goal = Point(60, uniform(50, 60))

        time.sleep(0.1)  # 시각적으로 에피소드 구분
