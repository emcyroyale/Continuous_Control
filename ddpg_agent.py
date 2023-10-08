import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic
from noise import OUNoise
from memory import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed, config, device):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.device = device
        self.config = config

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(self.device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(self.device)
        #self.actor_target.load_state_dict(self.actor_local.state_dict())
        #self.actor_target.parameters().data.copy_(self.actor_local.parameters().data)
        #for target_param, local_param in zip(self.actor_target.parameters(), self.actor_local.parameters()):
        #    target_param.data.copy_(local_param.data)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=self.config['lr_actor'])
        
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, random_seed).to(self.device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(self.device)
        #self.critic_target.load_state_dict(self.critic_local.state_dict())
        #self.critic_target.parameters().data.copy_(self.critic_local.parameters().data)
        #for target_param, local_param in zip(self.critic_target.parameters(), self.critic_local.parameters()):
        #    target_param.data.copy_(local_param.data)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.config['lr_critic'],
                                           weight_decay=self.config['weight_decay'])

        # Noise process
        self.noise = OUNoise((config['num_agents'], action_size),
                             self.seed,
                             theta=self.config['theta'])

        # Replay memory
        self.memory = ReplayBuffer(action_size,
                                   self.config['buffer_size'],
                                   self.config['batch_size'],
                                   self.seed,
                                   device)
        
        self.mean_learning_sample_reward = 0
        self.n_learning_samples = 0 
    
    def add_to_replay(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
    
    def step(self, states, actions, rewards, next_states, dones, add_replay_flag=True, learn_flag=True, learn_repeat=20):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        if add_replay_flag:
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                self.add_to_replay(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.config['batch_size'] and learn_flag:
            for i in range(learn_repeat):
                experiences = self.memory.sample()
                self.learn(experiences, self.config['gamma'])

    def act(self, state, add_noise=True, noise_weight=1.0):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(self.device)
        
        self.actor_local.eval()
        
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        
        self.actor_local.train()
        
        if add_noise:
            action += (noise_weight * self.noise.sample())
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        #self.mean_learning_sample_reward += np.sum(rewards.cpu().data.numpy())
        self.n_learning_samples += len(rewards)
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()
        
        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.config['tau'])
        self.soft_update(self.actor_local, self.actor_target, self.config['tau'])                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
