# Welcome to the wiki of my Bachelor Thesis.

I created this wiki primarily for myself. This project grew so complex in such a short time that I needed a place to write down thoughts, ideas but especially the **structure** of the project itself.

### TensorFlow and Reinforcement Learning
The main objective of this (wiki and) project is to create a **Reinforcement Learning** environment for the swarm simulation. The environment is based on the [TensorFlow](https://www.tensorflow.org/) framework, which is a very powerful tool for machine learning. It is used by many companies and research groups, including Google, DeepMind, OpenAI and many more.

**However**, as a beginner, implementing a RL environment with TensorFlow is not an easy task. Existing tutorials and examples have to be tailored to the specific use case. Unfortunately, the swarm simulation is a very complex environment due to its "Multi-Agent"-like nature. 

The goal is **not** to implement a multi-agent environment, but to create a single-agent environment that can be used to train a single agent to control the swarm. The agent will be able to control the swarm by sending commands to the individual agents. The individual agents will then execute the commands and send back the results to the agent. The agent will then be able to observe the results and learn from them.

This implementation needs a rigourous understanding of the pseudo-Multi-Agent-Reinforcement-Learning, the TensorFlow framework and the swarm simulation itself.


# Reinforcement Learning
# General
Reinforcement Learning is a subfield of (unsupervised) machine learning. It is based on the idea of learning by interacting with an environment. The agent is able to observe the environment and take actions. The environment will then return a reward and the next state. The agent will then learn from the reward and the next state. The goal is to maximize the reward.

### Unsupervised
Unlike supervised learning, the agent does not receive a label for each action. Instead, the agent has to learn from the reward and the next state. This is a much more difficult task since there is no "correct" answer.

### Markov Decision Process
Reinforcement Learning is based on the idea of a Markov Decision Process (MDP). A MDP is a tuple of the form `(S, A, P, R, γ)`, where:
- `S` is a set of states
- `A` is a set of actions
- `P` is a transition probability matrix
- `R` is a reward function
- `γ` is a discount factor

In Reinforcement Learning, the agent is in a state `s` and takes an action `a`. The environment will then return a reward `r` and the next state `s'`. The transition probability matrix `P` and reward function `R` are unknown to the agent. 
Instead, another function `Q` is used.

### Q-Function
The Q-Function `Q(s, a)` is a function that returns the expected reward for taking action `a` in state `s`. The goal of the agent is to maximize the Q-Function.

### State and Observation
A state is a representation of the environment. It contains all the information the agent needs to take an action. In Reinforcement Learning, the agent is not able to observe the entire environment. Instead, the agent is only able to observe a part of the environment. This is called the observation. The observation is a subset of the state.

In the case of the swarm simulation, a ``state`` represents all particles with all their properties and all order parameters.

An ``observation`` would be the k-nearest neighbors of a particle. This is a subset of the state.

# Framework
# General
As mentioned, the framework used for this project is [TensorFlow](https://www.tensorflow.org/). 

For this project, a module and sub-framework called [TensorFlow Agents](https://www.tensorflow.org/agents) is used. TensorFlow Agents is a collection of Reinforcement Learning algorithms implemented in TensorFlow.

## TensorFlow Agents
There are many Reinforcement Learning algorithms. TensorFlow Agents implements the most common ones. The Agents (and their respective algorithms) differ primarily in what types of problems they can solve.

### Discrete vs. Continuous
Some environments have a discrete action space (e.g. an Agent that can move left, right, up or down for a fixed distance; think of games), while others have a continuous action space (e.g. in this project: the Agent can change the angle of a particle continuously).

As with the environments, some algorithms can only solve discrete problems, while others can solve (only) continuous problems. TensorFlow provides a `wrapper` to disctretize continuous action spaces, although coming with a loss of accuracy as trade-off.

# Overview of the Agents
![image](https://github.com/RnLe/bachelor_thesis23/assets/34630928/d6c6b86b-4465-4b50-8e80-fbad1e0e2665)
In this project, a `DDPG`-Agent is used. The `DDPG`-Agent is a `ContinuousActorCriticAgent`. This means that the agent is able to solve continuous problems and uses an actor-critic algorithm.

## Actor-Critic
An actor-critic based Agent consists of two neural networks: an actor and a critic. The actor is responsible for taking actions, while the critic is responsible for learning the Q-Function.

They differ in their *input* and *output* (and in this case, in their complexity).
- The actor takes the observation as input and returns an action. This is the policy `π` of the agent, which is trained to maximize the Q-Function.
- The critic takes the observation **_AND_** the action as input and returns the Q-Value. This is the value function of the agent, which is trained to minimize the loss.

This can be interpreted as follows: The actor is part of the Agent, while the critic is part of the environment and provides feedback to the agent.

### Actor-Network
image
This network is fairly simple and represents the policy `π` of the agent. The aim of this project is to create a policy that is able to control the swarm with as little complexity as possible.

### Critic-Network
image
This network is more complex and represents the value function of the agent. The complex structure is necessary to learn the Q-Function. This network is only used during training.

# Overview of all components
The `DDPG`-Agent is only one component of the entire project and embedded in a larger framework. The following image shows the entire framework and how the components interact with each other.

image

The main components are:
- `Environment`
- `Agent`
- `Policies`
- `Neural Networks`
- `Replay Buffer`
- `Driver`
- `Actors`

Some minor components are:
- `Checkpointer and PolicySaver`
- `Learner`
- `Observer`
- `Strategy`