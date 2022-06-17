import gym 
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

environment_name = "Breakout-v4"
#env = gym.make(environment_name, render_mode='human')
env = make_atari_env(environment_name, n_envs=1, seed=0)
env = VecFrameStack(env, n_stack=4)
env.render_mode='human'
a2c_path = os.path.join('AtariRL', 'Saved Models', 'A2C_model3')
model = A2C.load(a2c_path, env=env)
print(evaluate_policy(model, env, n_eval_episodes=10, render=True))
#episodes = 10
#for episode in range(1, episodes+1):
#    obs = env.reset()
#    done = False
#    score = 0 
#    
#    while not done:
#        env.render()
#        action, states = model.predict(obs)
#        obs, reward, done, info = env.step(action)
#        score+=reward
#    print('Episode:{} Score:{}'.format(episode, score))
#env.close()

