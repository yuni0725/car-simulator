import numpy as np
from world import World
from agents import (
    Car,
    RectangleBuilding,
    Pedestrian,
    Painting,
    RectangleEntity,
    CircleEntity,
)
from geometry import Point
import time
from fast_lidar import calculate_lidar_distances

human_controller = True

dt = 0.1  # time steps in terms of seconds. In other words, 1/dt is the FPS.
w = World(
    dt, width=120, height=120, ppm=6
)  # The world is 120 meters by 120 meters. ppm is the pixels per meter.

# Let's add some sidewalks and RectangleBuildings.
# A Painting object is a rectangle that the vehicles cannot collide with. So we use them for the sidewalks.
# A RectangleBuilding object is also static -- it does not move. But as opposed to Painting, it can be collided with.
# For both of these objects, we give the center point and the size.

w.add(RectangleBuilding(Point(20, 60), Point(50, 180)))
w.add(RectangleBuilding(Point(100, 60), Point(50, 180)))


for i in range(11):
    w.add(Painting(Point(60, 12 * i), Point(0.5, 2), "white"))


def random_car_y():
    return uniform(10, 45)


# A Car object is a dynamic object -- it can move. We construct it using its center location and heading angle.
c1 = Car(Point(60, 0), np.pi / 2)
w.add(c1)

w.render()  # This visualizes the world we just constructed.

from random import uniform, randint

c2 = Car(Point(50, random_car_y()), np.pi / 2, "blue")
c2.velocity = Point(uniform(0, 5), 0)

w.add(c2)

c3 = Car(Point(60, random_car_y()), np.pi / 2, "green")
c3.velocity = Point(uniform(0, 5), 0)

w.add(c3)

c4 = Car(Point(70, random_car_y()), np.pi / 2, "yellow")
c4.velocity = Point(uniform(0, 5), 0)

w.add(c4)


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


while True:
    from interactive_controllers import KeyboardController

    controller = KeyboardController(w)
    for k in range(400):
        w.dynamic_agents[0].set_control(controller.steering * 10, controller.throttle)

        w.tick()  # This ticks the world for one time step (dt second)
        w.render()
        normalized_lidar = normalize_lidar(lidar_data(w.dynamic_agents[0], w))

        time.sleep(dt / 4)

        if w.collision_exists():
            print("Collision detected!")
            import sys

            sys.exit(0)

        movement_car = w.dynamic_agents[1]
        if movement_car.center.y > 65:
            w.dynamic_agents.remove(movement_car)
            new_car = Car(
                Point(movement_car.center.x, random_car_y()),
                heading=movement_car.heading,
                color=movement_car.color,
            )
            new_car.velocity = movement_car.velocity
            w.dynamic_agents.append(new_car)

        if w.dynamic_agents[0].center.y > 65:
            w.dynamic_agents.remove(w.dynamic_agents[0])
            new_target_car = Car(
                Point(60, 0),
                heading=np.pi / 2,
                color="red",
            )
            w.dynamic_agents.insert(0, new_target_car)

            for movement_car in w.dynamic_agents[1:]:
                w.dynamic_agents.remove(movement_car)
                new_car = Car(
                    Point(movement_car.center.x, random_car_y()),
                    heading=movement_car.heading,
                    color=movement_car.color,
                )

                new_car.velocity = movement_car.velocity
                w.dynamic_agents.append(new_car)
