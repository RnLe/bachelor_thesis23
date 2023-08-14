# %%
# Basics
from Solver import Particle, Perceptron, PerceptronModel, VicsekModel, NeuralNetwork, PerceptronMode, Mode, NeuralSwarmModel

import tensorflow   as tf
import numpy        as np
import os
import logging
import time
import matplotlib.pyplot as plt

# Logging
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import ray
from ray import rllib
from ray import tune
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms import maddpg

import gym
from gym import spaces
from gym.spaces import Box


# %% [markdown]
# # Simulation Parameters

# %%
# Simulation settings
settings = {
        #                  N,      L,      v,      noise,  r
        "small": [         100,    10,     0.03,   0.1,    1],
        "medium": [        1000,   10,     0.03,   0.1,    1],
    }
    
# Choose between RADIUS, FIXED, FIXEDRADIUS (don't use RADIUS)
mode = Mode.FIXEDRADIUS
# Flags
ZDimension = False     # 2D or 3D
seed = False           # Random seed
# Choose settings
chosen_settings = settings["small"]
N       = chosen_settings[0]
L       = chosen_settings[1]
v       = chosen_settings[2]
noise   = chosen_settings[3]
r       = chosen_settings[4]

k_neighbors = 5
# Timesteps in an episode
T = 10

# %% [markdown]
# # Custom Single-Agent Environment

# %%
# class SimulationGymEnvironment(gym.Env):
#     """A custom Gym environment for the swarm simulation."""
#     minimum = 0.0
#     maximum = 2 * np.pi

#     def __init__(self):
#         super(SimulationGymEnvironment, self).__init__()

#         # Action space: the angle of the particle
#         self.action_space = spaces.Box(low=self.minimum, high=self.maximum, shape=(), dtype=np.float32)

#         # Observation space: angles of k_neighbors + 1 particles
#         self.observation_space = spaces.Box(low=self.minimum, high=self.maximum, shape=(k_neighbors + 1,), dtype=np.float32)

#         # Initialization similar to the previous environment
#         self.simulation = NeuralSwarmModel(N, L, v, noise, r, mode, k_neighbors, ZDimension, seed=seed)
#         self.index = 0
#         self.new_angles = np.zeros(shape=(N,), dtype=np.float32)

#     def reset(self):
#         # The reset logic is quite similar to the previous environment
#         self.simulation = NeuralSwarmModel(N, L, v, noise, r, mode, k_neighbors, ZDimension, seed=False)
#         self.index = 0
#         observation = self.simulation.get_angles(self.index)
#         return observation

#     def step(self, action):
#         # Again, this is quite similar to the previous environment's _step method
#         action = np.clip(action, self.minimum, self.maximum)
#         initial_order_param = self.simulation.get_local_order_parameter(self.index)
#         self.new_angles[self.index] = action
#         self.simulation.update_angle(self.index, action)
#         final_order_param = self.simulation.get_local_order_parameter(self.index)
#         reward = final_order_param - initial_order_param
#         self.index += 1
#         observation = self.simulation.get_angles(self.index)
#         done = False
#         if self.index >= N - 1:
#             self.index = 0
#             self.simulation.update()
#             done = True
#         return observation, reward, done, {}  # The empty dict is for 'info', which we aren't using

#     def render(self, mode='human'):
#         # If you have any visualization code, it would go here.
#         pass

#     def close(self):
#         # Any cleanup code, if necessary.
#         pass


# %% [markdown]
# # Custom Multi-Agent Environment

# %%
class MultiAgentSimulationEnv(MultiAgentEnv):
    minimum = 0.0
    maximum = 2 * np.pi
    
    def __init__(self):
        self.num_agents = N
        
        # We asume the same action space for all agents
        self.action_space = Box(low=self.minimum, high=self.maximum, shape=(), dtype=np.float32)
        
        # We assume the same observation space for all agents
        self.observation_space = Box(low=self.minimum, high=self.maximum, shape=(k_neighbors + 1,), dtype=np.float32)
        
        self.simulation = NeuralSwarmModel(N, L, v, noise, r, mode, k_neighbors, ZDimension, seed=seed)
        self.new_angles = np.zeros(shape=(N,), dtype=np.float32)
        self.index = 0

    def reset(self):
        # Reset the state of the environment to an initial state
        observations = {}
        self.simulation = NeuralSwarmModel(N, L, v, noise, r, mode, k_neighbors, ZDimension, seed=False)
        self.index = 0
        self.new_angles = np.zeros(shape=(N,), dtype=np.float32)
        for agent_id in range(self.num_agents):
            observations[agent_id] = self.simulation.get_angles(agent_id)
        return observations

    def step(self, action_dict):
        # Actions for all agents are provided in a dictionary
        rewards = {}
        new_obs = {}
        dones = {}
        info = {}
        
        # Collect all actions and set dones
        for agent_id, action in action_dict.items():
            action = np.clip(action, self.minimum, self.maximum)
            self.new_angles[agent_id] = action
            dones[agent_id] = True if self.index >= T else False
            
        # Update the simulation
        self.simulation.update_angles(self.new_angles)
        self.simulation.update()
        self.index += 1
        reward = self.simulation.mean_direction2D()
        
        # Collect observations and rewards
        for agent_id in range(self.num_agents):
            new_obs[agent_id] = self.simulation.get_angles(agent_id)
            rewards[agent_id] = reward

        dones['__all__'] = all(dones.values())  # Ends the episode if all agents are done
        
        return new_obs, rewards, dones, info

    def render(self, mode='human'):
        # Optional: For visualization
        # Draw particles with matplotlib
        # Particles are stored in self.simulation.particles . Positions are stored in particles[i].x and particles[i].y
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Simulation')
        
        for particle in self.simulation.particles:
            ax.plot(particle.x, particle.y, 'o', color='black', markersize=10)
            
        plt.show()

    def close(self):
        # Optional: Clean up. Called at the end of an episode.
        pass

# %% [markdown]
# # Create environment

# %%
env = MultiAgentSimulationEnv()

for i_episode in range(1):
    observations = env.reset()
    total_rewards = {agent_id: 0 for agent_id in observations.keys()}
    print(f"Starting episode {i_episode}")
    
    # Max steps per episode
    for t in range(T + 1):
        # Optional: Render the environment for visualization
        env.render()
        
        # Choose random actions
        actions = {agent_id: env.action_space.sample() for agent_id in observations.keys()}
        
        observations, rewards, dones, infos = env.step(actions)
        
        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward
            
        print(f"Step {t}... \r", end="")
            
        if any(dones.values()):
            print(f"Step {t} finished")
            print(f"Episode {i_episode} finished after {t} timesteps with rewards: {next(iter(rewards.values()))}")
            break

env.close()

# %% [markdown]
# # Policy Mapping
# 
# In the following code, ``policy_mapping_fn(agent_id)`` is defined to map each agent to a policy. The agents id is used to map each agent to a policy. The policy is then used to compute the action for each agent.
# 
# In this case, a shared policy is used for all agents. The policy is defined in the ``policy_graph`` function. The policy is a simple neural network.
# 
# 

# %%
def policy_mapping_fn(agent_id):
    """Returns the policy that should be used by the agent with the id agent_id.
    In this case, all agents share the same policy."""
    return "shared_policy"

# %% [markdown]
# # Configurations

# %%
config = {
    "multiagent": {
        "policies": {
            "shared_policy": (None, env.observation_space, env.action_space, {
                "model": {
                    "fcnet_hiddens": [],  # Hidden, fully-connected layers in the model (e.g. [64, 128])
                },
            }),
        },
        "policy_mapping_fn": policy_mapping_fn,
    },
    "framework": "tf"
    # ... additional config
}


