import numpy as np
from numba import njit


@njit(cache=True)
def _point_in_rotated_rect(px, py, rect_x, rect_y, rect_w, rect_h, rect_angle):
    """
    Check if a point is inside a rotated rectangle.
    """
    # Translate point to rectangle's local coordinate system
    local_px = px - rect_x
    local_py = py - rect_y

    # Rotate the point by the negative of the rectangle's angle
    cos_angle = np.cos(-rect_angle)
    sin_angle = np.sin(-rect_angle)

    rotated_px = local_px * cos_angle - local_py * sin_angle
    rotated_py = local_px * sin_angle + local_py * cos_angle

    # Check if the rotated point is within the axis-aligned rectangle
    half_w = rect_w / 2.0
    half_h = rect_h / 2.0

    return (-half_w <= rotated_px <= half_w) and (-half_h <= rotated_py <= half_h)


@njit(cache=True)
def _point_in_circle(px, py, circle_x, circle_y, circle_r):
    """
    Check if a point is inside a circle.
    """
    return (px - circle_x) ** 2 + (py - circle_y) ** 2 < circle_r**2


@njit(cache=True)
def calculate_lidar_distances(
    car_pos_x,
    car_pos_y,
    car_heading,
    rect_obstacles,
    circle_obstacles,
    num_rays=90,
    max_distance=50.0,
    fov_deg=180.0,
    num_steps=200,
):
    """
    Numba-accelerated Lidar calculation.
    """
    half_fov_rad = np.deg2rad(fov_deg / 2)
    angles = np.linspace(-half_fov_rad, half_fov_rad, num_rays)

    lidar_distances = np.full(num_rays, max_distance, dtype=np.float32)

    for i in range(num_rays):
        ray_angle = car_heading + angles[i]

        for d in np.linspace(0.1, max_distance, num_steps):
            px = car_pos_x + d * np.cos(ray_angle)
            py = car_pos_y + d * np.sin(ray_angle)

            collision_found = False

            # Check for collisions with rectangular obstacles
            for j in range(rect_obstacles.shape[0]):
                rect = rect_obstacles[j]
                if _point_in_rotated_rect(
                    px, py, rect[0], rect[1], rect[2], rect[3], rect[4]
                ):
                    lidar_distances[i] = d
                    collision_found = True
                    break

            if collision_found:
                break

            # Check for collisions with circular obstacles
            for j in range(circle_obstacles.shape[0]):
                circle = circle_obstacles[j]
                if _point_in_circle(px, py, circle[0], circle[1], circle[2]):
                    lidar_distances[i] = d
                    collision_found = True
                    break

            if collision_found:
                break

    return lidar_distances


def normalize_lidar(lidar_array, max_value=50.0):
    """
    lidar_array: 1D or 2D numpy array
    max_value: 정규화 기준 최대값
    return: 0~1 사이로 정규화된 배열
    """
    lidar_array = np.clip(lidar_array, 0, max_value)
    return lidar_array / max_value
