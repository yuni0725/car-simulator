from stable_baselines3 import PPO
from environment import CarSimulations  # ✅ 네 환경 클래스

from stable_baselines3.common.env_util import make_vec_env

# 환경 생성 (주의: 학습할 땐 VecEnv 형태로)
env = make_vec_env(CarSimulations, n_envs=1)

# 저장된 모델 로드
model = PPO.load("car_sim_400000", env=env)

# 이후 이어서 학습할 수 있음
model.learn(total_timesteps=100000)
