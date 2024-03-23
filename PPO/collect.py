#from agent import Agent
import torch
import argparse
import pickle
from stable_baselines3 import PPO

from farmgym_games.game_builder.utils_sb3 import farmgym_to_gym_observations_flattened, wrapper
from farmgym_games.game_catalogue.farm0.farm import env as Farm0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = Farm0()
env.farmgym_to_gym_observations = farmgym_to_gym_observations_flattened
env = wrapper(env)

def collect(args):
    agent = PPO.load("Heuristic_Agent")
    #agent.load_weights()

    # dict of arrays
    memory = {'states': [], 'actions': [], 'rewards': [], 'terminals': []}

    rewards = []
    trajectories = 0

    while trajectories < args.n_traj:
        ep_reward = 0
        state, _ = env.reset()
        reward = 0
        action = 0
        terminal = False

        ep_memory = {'state': [], 'action': [], 'reward': [], 'terminal': []}
        
        while True:
            #prob_a = agent.policy_network.pi(torch.FloatTensor(state).to(device))
            #action = torch.distributions.Categorical(prob_a).sample().item()
            action, _ = agent.predict(state)
            new_state, reward, terminal, info, _ = env.step(action)
            ep_reward += reward

            ep_memory['state'].append(state)
            ep_memory['action'].append(action)
            ep_memory['reward'].append(reward)
            ep_memory['terminal'].append(terminal)
            state = new_state

            if terminal:
                if ep_reward >= args.min_reward:
                    rewards.append(ep_reward)
                    memory['states'] += ep_memory['state']
                    memory['actions'] += ep_memory['action']
                    memory['rewards'] += ep_memory['reward']
                    memory['terminals'] += ep_memory['terminal']
                    trajectories += 1
                    print("trajectory reward: {}, collected {} trajectories".format(ep_reward, trajectories))
                break

    #agent.env.close()
    avg_rew = sum(rewards) / len(rewards)
    print('avg rew: %.2f' % avg_rew)
    print('trajectories:', trajectories)
    print('states collected:', len(memory['states']))

    f = open(args.traj_path, 'wb')
    pickle.dump(memory, f)
    f.close()
    print("trajectories saved to", args.traj_path)
