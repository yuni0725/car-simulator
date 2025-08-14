import gym
from gym import spaces
import numpy as np
import time

from simulations.world import World

from simulations.agents import (
    Car,
    Painting,
    RectangleBuilding,
    RectangleEntity,
    CircleEntity,
)
from simulations.geometry import Point
from random import uniform, randint
from fast_lidar import calculate_lidar_distances


def random_car_y():
    """Generate random y position for cars"""
    return uniform(60, 60)


def random_car_velocity():
    return uniform(1, 3)


class CarSimulations(gym.Env):

    metadata = {"render_fps": 60}

    def __init__(self):
        super(CarSimulations, self).__init__()

        self.reward_range = (-200, 100)
        self.goal_position = Point(
            60, uniform(110, 120)
        )  # Goal at the top of the track

        self.observation_space = spaces.Dict(
            {
                "lidar_data": spaces.Box(
                    low=0, high=1.0, shape=(90,), dtype=np.float32
                ),
                "velocity": spaces.Box(low=0, high=10, shape=(1,), dtype=np.float32),
                "steering": spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32),
            }
        )

        self.speed_values = [0, 5, 10]
        self.steering_values = [-10, -5, 0, 5, 10]
        self.action_map = [
            (sp, st) for sp in self.speed_values for st in self.steering_values
        ]

        self.action_space = spaces.Discrete(len(self.action_map))

        self.dt = (
            0.1  # time steps in terms of seconds. In other words, 1/dt is the FPS.
        )
        world_width = 120  # in mm
        world_height = 120

        self.w = World(self.dt, width=world_width, height=world_height, ppm=6)

        # World setup from example-random.py
        # self.w.add(RectangleBuilding(Point(10, 60), Point(30, 120)))
        # self.w.add(RectangleBuilding(Point(110, 60), Point(30, 120)))
        self.w.add(RectangleBuilding(Point(20, 60), Point(50, 180)))
        self.w.add(RectangleBuilding(Point(100, 60), Point(50, 180)))

        for i in range(11):
            self.w.add(Painting(Point(60, 12 * i), Point(0.5, 2), "white"))

        # This is the agent-controlled car
        self.car = None
        # These are the other cars
        self.other_cars = []

        # Car configurations: (x_position, color)
        self.other_car_configs = [(50, "blue"), (60, "green"), (70, "yellow")]
        # self.other_car_configs = [(60, "blue")]

        self.reset()

    def _create_other_cars(self):
        """Create multiple other cars at different x coordinates"""
        self.other_cars = []
        for x_pos, color in self.other_car_configs:
            car = Car(Point(x_pos, random_car_y()), np.pi / 2, color)
            car.velocity = Point(random_car_velocity(), 0)
            self.other_cars.append(car)
            self.w.add(car)

    def add_car_config(self, x_position, color):
        """Easily add a new car configuration"""
        self.other_car_configs.append((x_position, color))

    def reset(self):
        self.w.reset()  # Clears dynamic agents

        # Add agent's car
        self.car = Car(Point(60, 0), np.pi / 2)
        self.car.max_speed = 10.0
        self.car.velocity = Point(0, 0)
        self.w.add(self.car)

        # Add other cars
        self._create_other_cars()

        self.prev_position = self.car.center
        self.dist_to_goal = np.linalg.norm(
            [
                self.car.center.x - self.goal_position.x,
                self.car.center.y - self.goal_position.y,
            ]
        )
        self.x_dist_to_goal = abs(self.car.center.x - self.goal_position.x)
        self.acceleration = 0.0
        self.velocity = 0.0
        self.steering = 0
        self.stagnant_frames = 0
        self.current_step = 0

        self.w.render()

        lidar_data = self._get_lidar_data()
        norm_lidar = self.normalize_lidar(lidar_data)

        return {
            "lidar_data": norm_lidar.astype(np.float32),
            "velocity": np.array([self.velocity], dtype=np.float32),
            "steering": np.array([self.steering], dtype=np.float32),
        }

    def _get_lidar_data(self, num_rays=90, max_distance=50.0, fov_deg=180.0):
        rect_obstacles = []
        circle_obstacles = []

        for agent in self.w.agents:
            if not agent.collidable or agent is self.car:
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
            self.car.center.x,
            self.car.center.y,
            self.car.heading,
            rects_np,
            circles_np,
            num_rays=num_rays,
            max_distance=max_distance,
            fov_deg=fov_deg,
        )

    def normalize_lidar(self, lidar: np.ndarray, max_range: float = 50.0) -> np.ndarray:
        """
        라이다 거리 데이터를 0~1 사이로 정규화.
        max_range보다 큰 값은 1.0으로 클리핑됨.
        """
        lidar = np.clip(lidar, 0, max_range)
        return lidar / max_range

    def step(self, action_index):
        # 1. Take action
        self.velocity, self.steering = self.action_map[action_index]

        self.car.set_control(self.steering, 0.0)

        self.car.velocity = Point(0, int(self.velocity))

        self.w.tick()
        self.w.render()
        self.current_step += 1

        done = False
        reward = 0.0

        if self.car.center.y > 115:
            self.w.dynamic_agents.remove(self.car)
            self.car = Car(Point(60, 0), heading=np.pi / 2, color="red")
            self.w.add(self.car)

            # Also respawn all other cars
            for other_car in self.other_cars:
                if other_car in self.w.dynamic_agents:
                    self.w.dynamic_agents.remove(other_car)

            self._create_other_cars()

        # Handle other cars respawn when they go off-screen
        for i, other_car in enumerate(self.other_cars):
            if other_car.center.y > 115:
                if other_car in self.w.dynamic_agents:
                    self.w.dynamic_agents.remove(other_car)

                # Create new car with same configuration
                x_pos, color = self.other_car_configs[i]
                new_car = Car(Point(x_pos, random_car_y()), np.pi / 2, color)
                new_car.velocity = Point(random_car_velocity(), 0)
                self.other_cars[i] = new_car
                self.w.add(new_car)

        speed = self.car.speed
        reward += speed * 0.02

        reward += self.x_dist_to_goal * 0.1

        curr_pos = self.car.center
        self.dist_to_goal = np.linalg.norm(
            [
                curr_pos.x - self.goal_position.x,
                curr_pos.y - self.goal_position.y,
            ]
        )

        reward -= abs(self.steering) * 0.005

        dist_moved = np.linalg.norm(
            [curr_pos.x - self.prev_position.x, curr_pos.y - self.prev_position.y]
        )
        if dist_moved < 0.1 and self.acceleration < 0.1:
            self.stagnant_frames += 1
            reward -= 0.2
        else:
            self.stagnant_frames = 0
        self.prev_position = curr_pos

        if self.w.collision_exists(self.car):
            reward = -20.0
            done = True
        elif self.dist_to_goal < 5:
            reward += 10.0
            done = True
            print("Reached the goal!")

        if self.stagnant_frames >= 100:
            reward -= 1.0
            done = True

        if self.current_step >= 3000:
            done = True

        # 5. Get next observation
        lidar_data = self._get_lidar_data()
        norm_lidar = self.normalize_lidar(lidar_data)

        obs = {
            "lidar_data": norm_lidar.astype(np.float32),
            "velocity": np.array([self.velocity], dtype=np.float32),
            "steering": np.array([self.steering], dtype=np.float32),
        }

        return obs, reward, done, {}

    def render(self):
        self.w.render()

    def close(self):
        self.w.close()
