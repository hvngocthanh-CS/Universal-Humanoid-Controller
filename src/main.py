from ddpg_torch import Agent
import gym
import numpy as np
from utils import plotLearning
import os

def play_multiple_times(agent, env, time_steps, game, save_freq=25, random_seed=42, type='train'):
    np.random.seed(random_seed)
    score_history = []
    
    if type.lower() != 'test':
        game_dir = os.path.join('models', game)
        filename = game + '-alpha000025-beta00025-400-300.png'
    
    if type.lower() != 'train':
        agent.load_models()
        
    for i in range(time_steps):
        obs, _ = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, terminated, truncated, _ = env.step(act)
            if type.lower() != 'test':
                agent.remember(obs, act, reward, new_state, int(terminated))
                agent.learn()
            score += reward
            obs = new_state
            env.render()
            done = terminated or truncated
        if type != 'test':
            print('episode ', i, 'score %.2f' % score)
            score_history.append(score)
        
        if type.lower() != 'test' and i % save_freq == 0:
            agent.save_models()

            with open(os.path.join(game_dir, 'scores.txt'), 'a') as f:
                for score in score_history:
                    f.write(str(score) + '\n')
            score_history = []
        
        if type.lower() == 'test':
            print('episode ', i, 'score %.2f' % score)
            
    return score_history

if __name__ == '__main__':
    game = 'Humanoid-v4'
    type = 'train'
    model_path = os.path.join('./models', game)
    ep=10
    if type.lower() != 'test':
        env = gym.make(game, render_mode=None)
    else:
        env = gym.make(game, render_mode='human')
    agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[env.observation_space.shape[0]], tau=0.001, env=env,
                  batch_size=64,  layer1_size=400, layer2_size=300, n_actions=env.action_space.shape[0], 
                  model_path=model_path)

    score_history = play_multiple_times(agent, env, 100, game, save_freq=25, type=type)
    # if type.lower() == 'train':
    #     game_dir = os.path.join('models', game)
    #     filename = game + '-alpha000025-beta00025-400-300.png'
    #     plotLearning(score_history, os.path.join(game_dir, filename), window=100)