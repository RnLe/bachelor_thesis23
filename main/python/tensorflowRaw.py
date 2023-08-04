from Solver import Particle, Perceptron, PerceptronModel, VicsekModel, NeuralNetwork, PerceptronMode, Mode, NeuralSwarmModel

import tensorflow as tf
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Simulation settings
settings = {
        #                  N,      L,      v,      noise,  r
        "XXsmall": [       5,      4,      0.03,   0.1,    1],
        "Xsmall": [        20,     6,      0.03,   0.1,    1],
        "small": [         100,    30,     0.03,   0.1,    1],
        "a": [             300,    7,      0.03,   2.0,    1],
        "b": [             300,    25,     0.03,   0.5,    1],
        "d": [             300,    5,      0.03,   0.1,    1],
        "plot1_N40": [     40,     3.1,    0.03,   0.1,    1],
        "large": [         2000,   60,     0.03,   0.3,    1]
    }
    
# Choose between RADIUS, FIXED, FIXEDRADIUS
mode = Mode.FIXEDRADIUS
# Flags
ZDimension = False     # 2D or 3D
# Duration of simulation
timesteps = 5000
# Choose settings
chosen_settings = settings["small"]
N       = chosen_settings[0]
L       = chosen_settings[1]
v       = chosen_settings[2]
noise   = chosen_settings[3]
r       = chosen_settings[4]
k_neighbors = 5

# Use "num_iterations = 1e6" for better results (2 hrs)
# 1e5 is just so this doesn't take too long (1 hr)
num_iterations = 10000 # @param {type:"integer"}

initial_collect_steps = 1000 # @param {type:"integer"}
collect_steps_per_iteration = 1 # @param {type:"integer"}
replay_buffer_capacity = 1000 # @param {type:"integer"}

batch_size = 32 # @param {type:"integer"}

critic_learning_rate = 3e-4 # @param {type:"number"}
actor_learning_rate = 3e-4 # @param {type:"number"}
alpha_learning_rate = 3e-4 # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}

actor_fc_layer_params = (256, 256)
critic_joint_fc_layer_params = (256, 256)

log_interval = 500 # @param {type:"integer"}

num_eval_episodes = 20 # @param {type:"integer"}
eval_interval = 1000 # @param {type:"integer"}

policy_save_interval = 500 # @param {type:"integer"}

from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment


# Create a custom environment
class SimulationEnvironment(py_environment.PyEnvironment):
    """Interface for a swarm simulation environment.
    
    Can be converted into a TensorFlow environment.
    
    Provides uniform access to the simulation and hosts the reward function.
    """
    
    def __init__(self):
        # THOUGHTS: This should be correct. A particle can only choose an angle, which is a float (scalar)
        # Allow for small negative values to allow for numerical errors
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.float32, minimum=-1e-6, maximum=2*np.pi, name='action')
        
        # THOUGHTS: This should be correct. The observation is a vector of length k_neighbors + 1, where each entry is an angle. There is no information about the position of the particles.
        
        # k_neighbors + 1 because the particle itself is also included
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(k_neighbors + 1,), dtype=np.float32, minimum=0, maximum=2*np.pi, name='observation')
        # THOUGHTS: What is _state? Is it the current state of the environment?
        # This should be different from observation, because the observation is what the agent sees, while the state is the actual state of the environment.
        # So the state should be the order parameter which is to be maximized.
        self._episode_ended = False
        self.simulation = NeuralSwarmModel(N, L, v, noise, r, mode, k_neighbors, ZDimension, seed=True)
        self._state = self.simulation.mean_direction2D()
        # One "episode" and its corresponding reward consists of the iteration over all N particles
        self.index = 0
        # To change all angles at once, we need to store the new angles in a list
        self.new_angles = np.zeros(shape=(N,), dtype=np.float32)
        observation = self.simulation.get_angles(self.index)
        self._current_time_step = ts.restart(np.array(observation, dtype=np.float32))

    def observation_spec(self):
        """Return observation_spec."""
        # DONE
        return self._observation_spec

    def action_spec(self):
        """Return action_spec."""
        # DONE
        return self._action_spec
    
    def reset(self):
        """Return initial_time_step and reset the simulation.
        
        Note that this is a hard reset and not a reset for the current epoch."""
        
        # DONE
        self._current_time_step = self._reset()
        return self._current_time_step

    def step(self, action):
        """Apply action and return new time_step."""
        # DONE
        if self._current_time_step is None:
            return self.reset()
        self._current_time_step = self._step(action)
        return self._current_time_step

    def current_time_step(self):
        # DONE
        return self._current_time_step

    # def time_step_spec(self):
        """Return time_step_spec."""
        # DONE
        # return ts.time_step_spec(self.observation_spec())

    def _reset(self):
        """Return initial_time_step and reset the simulation.
        
        Note that this is a hard reset and not a reset for the current epoch."""
        # THOUGHTS: In this case, a differentiation has to be made between an episode and an epoch.
        # The episode ends when all particles have been updated. An epoch ends when the simulation is reset.
        
        # Reset simulation
        self.simulation = NeuralSwarmModel(N, L, v, noise, r, mode, k_neighbors, ZDimension, seed=True)
        self._state = self.simulation.mean_direction2D()
        self._episode_ended = False
        self.index = 0
        observation = self.simulation.get_angles(self.index)
        return ts.restart(np.array(observation, dtype=np.float32))

    def _step(self, action):
        """Apply action and return new time_step.
        This method hosts the reward function."""

        # if self._episode_ended:
        #     # The last action ended the episode. Ignore the current action and start a new episode.
        #     return self.reset()

        # TODO: Make sure episodes don't go on forever. Define a stopping action.
        if action >= 0. and self._episode_ended is False:
            # Update angle of the particle, but don't update the simulation yet
            self.new_angles[self.index] = action    
        elif self._episode_ended is False:
            raise ValueError('What did you do? This should be a finite float value.')
        
        # Properly handle the case when the episode ends:
        # [x] The episode ends when all particles have been updated
        # [x] The driver needs to work with the updated simulation. So the simulation needs to go on instead of being reset (after self._episode_ended = True)

        if self._episode_ended:
            # Update all angles at once
            self.simulation.update_angles(self.new_angles)
            
            oldState = self._state
            
            self.simulation.update()
            self._state = self.simulation.mean_direction2D()
            
            # The reward is the difference between the new state and the old state.
            # An increase in the order parameter is rewarded, a decrease is punished.
            reward = self._state - oldState
            observation = self.simulation.get_angles(self.index)
            # The observation (first argument of ts.termination) is the angles of the neighbors of the particle
            self.index = 0
            return ts.termination(np.array(observation, dtype=np.float32), reward)
        else:
            observation = self.simulation.get_angles(self.index)
            self.index += 1
            if self.index >= N:
                self._episode_ended = True
            return ts.transition(np.array(observation, dtype=np.float32), reward=0.0, discount=1.0)
        
from tf_agents.train.utils import strategy_utils

# Distribution strategy
# For now, don't use GPU or TPU

strategy = strategy_utils.get_strategy(tpu=False, use_gpu=True)

# All variables and Agents need to be created under strategy.scope()

from tf_agents.networks import network
from tf_agents.train.utils import spec_utils
from tf_agents.environments import tf_py_environment

# There are two networks and two environments: one for training and one for evaluation.
collect_env = SimulationEnvironment()
eval_env = SimulationEnvironment()

# Wrap the environment in a TF environment.
tf_collect_env = tf_py_environment.TFPyEnvironment(collect_env)
tf_eval_env = tf_py_environment.TFPyEnvironment(eval_env)

# For the network to work with the environment, the specs have to be known.
observation_spec, action_spec, time_step_spec = (spec_utils.get_tensor_specs(tf_collect_env))

from tf_agents.environments import utils

# Test the python environments

utils.validate_py_environment(collect_env, episodes=N)
utils.validate_py_environment(eval_env, episodes=N)

from tf_agents.agents.ddpg import critic_network
from tf_agents.networks import actor_distribution_network
from tf_agents.agents.sac import tanh_normal_projection_network

# Define a network that can learn to predict the action given an observation.
# This is a simple (on demand fully connected) network that takes in an observation and outputs an action.

class ActorNet(network.Network):

  def __init__(self, input_tensor_spec, output_tensor_spec):
    super(ActorNet, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=(),
        name='ActorNet')
    self._output_tensor_spec = output_tensor_spec
    # THOUGHTS: For now, only one layer is used. This can be changed later.
    self._sub_layers = [
        tf.keras.layers.Dense(
            action_spec.shape.num_elements(), activation="linear"),
    ]

  def call(self, observations, step_type, network_state):
    del step_type

    output = tf.cast(observations, dtype=tf.float32)
    for layer in self._sub_layers:
      output = layer(output)
    actions = tf.reshape(output, [-1] + self._output_tensor_spec.shape.as_list())

    # Scale and shift actions to the correct range if necessary.
    return actions, network_state


# Create the Actor Network
actor = ActorNet(
    input_tensor_spec=observation_spec,
    output_tensor_spec=action_spec)


# Critic Network
with strategy.scope():
  critic_net = critic_network.CriticNetwork(
        (observation_spec, action_spec),
        observation_fc_layer_params=None,
        action_fc_layer_params=None,
        joint_fc_layer_params=critic_joint_fc_layer_params,
        activation_fn=None,
        kernel_initializer='glorot_uniform',
        last_kernel_initializer='glorot_uniform')
  
# Actor Distribution Network
# This is a distribution over the actor nerwork
# TODO: It is not clear what continuous_projection_net does. tanh might not be the best choice.
with strategy.scope():
  actor_net = actor_distribution_network.ActorDistributionNetwork(
      observation_spec,
      action_spec,
      activation_fn=None,
      fc_layer_params=actor_fc_layer_params,
      continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork)
  
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.train.utils import train_utils

# Initialize the agent

with strategy.scope():
  train_step = train_utils.create_train_step()

  tf_agent = ddpg_agent.DdpgAgent(
        time_step_spec,
        action_spec,
        actor_network=actor_net,
        critic_network=critic_net,
        actor_optimizer=tf.keras.optimizers.Adam(
            learning_rate=actor_learning_rate),
        critic_optimizer=tf.keras.optimizers.Adam(
            learning_rate=critic_learning_rate),
        target_update_tau=target_update_tau,
        target_update_period=target_update_period,
        td_errors_loss_fn=tf.math.squared_difference,
        gamma=gamma,
        reward_scale_factor=reward_scale_factor,
        train_step_counter=train_step)

  tf_agent.initialize()
  
import reverb
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils

# Use Reverb, a framework for experience replay developed by DeepMind, to store and sample experience tuples for training.
# Using a samples_per_insert somewhere between 2 and 1000. This is a trade-off between the number of samples that can be drawn from the replay buffer and the number of times the replay buffer needs to be updated.
rate_limiter=reverb.rate_limiters.SampleToInsertRatio(samples_per_insert=3.0, min_size_to_sample=3, error_buffer=3.0)

table_name = 'uniform_table'
table = reverb.Table(
    table_name,
    max_size=replay_buffer_capacity,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1))

reverb_server = reverb.Server([table])

# Since the agent needs N steps of experience to make an update, the dataset will need to sample batches of N steps + 1 to allow the agent to learn from a complete transition.

reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
    tf_agent.collect_data_spec,
    sequence_length= 2,
    table_name=table_name,
    local_server=reverb_server)

# A dataset is created from the replay buffer to be fed to the agent for training. 
dataset = reverb_replay.as_dataset(
      sample_batch_size=batch_size, num_steps=2).prefetch(50)
experience_dataset_fn = lambda: dataset

from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy

# Policies
# Create policies from the agent

tf_eval_policy = tf_agent.policy
eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
  tf_eval_policy, use_tf_function=True)

tf_collect_policy = tf_agent.collect_policy
collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
  tf_collect_policy, use_tf_function=True)

# Random policy to sample from the environment
random_policy = random_py_policy.RandomPyPolicy(
  collect_env.time_step_spec(), collect_env.action_spec())

from tf_agents.train import actor
from tf_agents.metrics import py_metrics
from tf_agents.train import learner

import tempfile

tempdir = tempfile.gettempdir()

# As the Actors run data collection steps, they pass trajectories of (state, action, reward) to the observer, which caches and writes them to the Reverb replay system.

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  reverb_replay.py_client,
  table_name,
  sequence_length= 2,
  stride_length=1)

# We create an Actor with the random policy and collect experiences to seed the replay buffer with.

initial_collect_actor = actor.Actor(
  collect_env,
  random_policy,
  train_step,
  steps_per_run=initial_collect_steps,
  observers=[rb_observer])

initial_collect_actor.run()

# Instantiate an Actor with the collect policy to gather more experiences during training.

env_step_metric = py_metrics.EnvironmentSteps()

collect_actor = actor.Actor(
  collect_env,
  collect_policy,
  train_step,
  steps_per_run= 2,
  metrics=actor.collect_metrics(10),
  summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
  observers=[rb_observer, env_step_metric])

# Create an Actor which will be used to evaluate the policy during training.
# actor.eval_metrics(num_eval_episodes) to log metrics later.

eval_actor = actor.Actor(
  eval_env,
  eval_policy,
  train_step,
  episodes_per_run=num_eval_episodes,
  metrics=actor.eval_metrics(num_eval_episodes),
  summary_dir=os.path.join(tempdir, 'eval'),
)

from tf_agents.train import triggers

# Learners
# The Learner component contains the agent and performs gradient step updates to the policy variables using experience data from the replay buffer.
# After one or more training steps, the Learner can push a new set of variable values to the variable container.

saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

# Triggers to save the agent's policy checkpoints.
learning_triggers = [
    triggers.PolicySavedModelTrigger(
        saved_model_dir,
        tf_agent,
        train_step,
        interval=policy_save_interval),
    triggers.StepPerSecondLogTrigger(train_step, interval=1000),
]

agent_learner = learner.Learner(
  tempdir,
  train_step,
  tf_agent,
  experience_dataset_fn,
  triggers=learning_triggers,
  strategy=strategy)

"""
We instantiated the eval Actor with actor.eval_metrics above, which creates most commonly used metrics during policy evaluation:

- Average return. The return is the sum of rewards obtained while running a policy in an environment for an episode, and we usually average this over a few episodes.
- Average episode length.

We run the Actor to generate these metrics.
"""

def get_eval_metrics():
  eval_actor.run()
  results = {}
  for metric in eval_actor.metrics:
    results[metric.name] = metric.result()
  return results

metrics = get_eval_metrics()

def log_eval_metrics(step, metrics):
  eval_results = (', ').join(
      '{} = {:.6f}'.format(name, result) for name, result in metrics.items())
  print('step = {0}: {1}'.format(step, eval_results))

log_eval_metrics(0, metrics)

# The training loop involves both collecting data from the environment and optimizing the agent's networks.
# Along the way, we will occasionally evaluate the agent's policy to see how we are doing.

# Reset the train step
tf_agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = get_eval_metrics()["AverageReturn"]
returns = [avg_return]

for _ in range(num_iterations):
  # Training.
  collect_actor.run()
  loss_info = agent_learner.run(iterations=1)

  # Evaluating.
  step = agent_learner.train_step_numpy

  if eval_interval and step % eval_interval == 0:
    metrics = get_eval_metrics()
    log_eval_metrics(step, metrics)
    returns.append(metrics["AverageReturn"])

  if log_interval and step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

rb_observer.close()
reverb_server.stop()

import matplotlib.pyplot as plt

steps = range(0, num_iterations + 1, eval_interval)
plt.plot(steps, returns)
plt.ylabel('Average Return')
plt.xlabel('Step')
plt.ylim()

