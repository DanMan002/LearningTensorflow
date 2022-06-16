import gym 
import os
from os.path import exists
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = "CartPole-v0"
#env = gym.make(environment_name)

#episodes = 100
#for episode in range(1, episodes+1):
#    state = env.reset()
#    done = False
#    score = 0 
#    
#    while not done:
#        env.render()
#        action = env.action_space.sample()
#        n_state, reward, done, info = env.step(action)
#        score+=reward
#    print('Episode:{} Score:{}'.format(episode, score))
#env.close()
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_Model_Cartpole')
saved_Models = os.path.join('Training', 'Saved Models')
log_path = os.path.join('Training', 'Logs')
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
if(exists(PPO_path+".zip")):
    model = PPO.load(PPO_path, env=env)
else:    
    model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_path)
model.learn(total_timesteps=20000)
model.save(PPO_path)

#print(evaluate_policy(model, env, n_eval_episodes=10, render=True))

#episodes = 5
#for episode in range(1, episodes+1):
#    obs = env.reset()
#   done = False
#    score = 0 
#    
#    while not done:
#        env.render()
#        action, states = model.predict(obs)
#        obs, reward, done, info = env.step(action)
#        score+=reward
#    print('Episode:{} Score:{}'.format(episode, score))
#env.close()