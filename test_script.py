#reticulate::use_virtualenv("venv")
#reticulate::repl_python()

import gym
import gym_fishing

from stable_baselines3 import SAC

env = gym.make("fishing-v4")
model = SAC("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=10000)

model.save("fishing-v4-SAC-Michael")
