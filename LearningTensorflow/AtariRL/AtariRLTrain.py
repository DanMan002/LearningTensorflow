from os.path import exists
import gym 
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.env_util import make_vec_env
import os

environment_name = "Breakout-v4"
#env = gym.make(environment_name, render_mode='human')
#env = DummyVecEnv([lambda: env])
env = make_atari_env(environment_name, n_envs=4, seed=0)
env = VecFrameStack(env, n_stack=4)
a2c_path = os.path.join('AtariRL', 'Saved Models', 'A2C_model3')
print(a2c_path)
if(exists(a2c_path+".zip")):
    model = A2C.load(a2c_path, env=env)
    print("From existing!\n")
else:    
    model = A2C("CnnPolicy", env, verbose=1)
    print("New model!\n")
model.learn(total_timesteps=200000)
model.save(a2c_path)

