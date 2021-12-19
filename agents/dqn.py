
import os
import torch 
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from . import networks


class DQN:
    
    def __init__(
        self,
        input_ch: int,
        n_actions: int,
        double_dqn: bool = False,
        dueling_dqn: bool = False,
        enc_hidden_ch: int = 32,
        enc_fdim: int = 1024,
        q_hidden_dim: int = 1024,
        gamma: float = 0.99,
        eps_max: float = 1.0,
        eps_min: float = 0.01,
        eps_decay_steps: int = 500000,
        target_update_interval: int = 10000,
        learning_rate: float = 0.0001,
        max_grad_norm: float = 1.0
    ):
        self.step = 0
        self.gamma = gamma 
        self.n_actions = n_actions
        self.eps_max = eps_max 
        self.eps_min = eps_min
        self.eps = eps_max 
        self.eps_decay_rate = (eps_max - eps_min) / eps_decay_steps
        self.trg_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not dueling_dqn:
            self.online_q = networks.QNetwork(input_ch, enc_hidden_ch, enc_fdim, q_hidden_dim, n_actions).to(self.device)
            self.optim = optim.Adam(self.online_q.parameters(), lr=learning_rate)
            if double_dqn:
                self.target_q = networks.QNetwork(input_ch, enc_hidden_ch, enc_fdim, q_hidden_dim, n_actions).to(self.device)
                self.target_q.load_state_dict(self.online_q.state_dict())
        else:
            self.online_q = networks.DuelQNetwork(input_ch, enc_hidden_ch, enc_fdim, q_hidden_dim, n_actions).to(self.device)
            self.optim = optim.Adam(self.online_q.parameters(), lr=learning_rate)
            if double_dqn:
                self.target_q = networks.DuelQNetwork(input_ch, enc_hidden_ch, enc_fdim, q_hidden_dim, n_actions).to(self.device)
                self.target_q.load_state_dict(self.online_q.state_dict())
            
    def trainable(self, state=True):
        if state:
            self.online_q.train()
            self.target_q.train()
        else:
            self.online_q.eval()
            self.target_q.eval()
            
    def save(self, out_dir):
        state = {'step': self.step, 'eps': self.eps, 'model_state': self.online_q.state_dict()}
        torch.save(state, os.path.join(out_dir, 'ckpt.pth'))
        
    def load(self, ckpt_dir):
        file = os.path.join(ckpt_dir, 'ckpt.pth')
        if os.path.exists(file):
            state = torch.load(file, map_location=self.device)
            self.step = state['step'] 
            self.eps = state['eps']
            self.online_q.load_state_dict(state['model_state'])
            self.target_q.load_state_dict(state['model_state'])
        else:
            raise FileNotFoundError(f'Could not find model checkpoint at {ckpt_dir}')
            
    def update_epsilon(self):
        new_eps = self.eps_max - self.step * self.eps_decay_rate
        self.eps = max(new_eps, self.eps_min)
        
    def update_target_net(self):
        if self.step % self.trg_update_interval == 0:
            self.target_q.load_state_dict(self.online_q.state_dict())
            
    def preprocess_obs(self, obs):
        if np.asarray(obs).ndim == 3:
            obs = np.asarray(obs).transpose((2, 0, 1))
            obs = torch.from_numpy(obs).to(self.device) / 255.0
            obs = obs.unsqueeze(0)
        elif np.asarray(obs).ndim == 4:
            obs = np.asarray(obs).transpose((0, 3, 1, 2)) 
            obs = torch.from_numpy(obs).to(self.device) / 255.0
        return obs
            
    def select_action(self, obs):
        obs = self.preprocess_obs(obs)
        
        if random.uniform(0, 1) < self.eps:
            action = random.randint(0, self.n_actions-1)
        else:
            with torch.no_grad():
                action = self.online_q(obs)[0].argmax(-1).item()
        
        self.step += 1 
        self.update_epsilon()
        self.update_target_net()
        return action
    
    def select_agent_action(self, obs):
        obs = self.preprocess_obs(obs)
        with torch.no_grad():
            action = self.online_q(obs)[0].argmax(-1).item()
        return action
    
    def learn_from_memory(self, batch):
        obs, action, reward, next_obs, done = batch 
        obs = self.preprocess_obs(obs)
        next_obs = self.preprocess_obs(next_obs)
        action = torch.from_numpy(action).long().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        done = torch.from_numpy(done).long().to(self.device)
        
        with torch.no_grad():
            next_action = self.online_q(next_obs)[0].argmax(-1)
            next_values = self.target_q(next_obs)[0].gather(1, next_action.view(-1, 1)).squeeze(-1)
            q_trg = reward + (1-done) * self.gamma * next_values 
            
        q_pred = self.online_q(obs)[0].gather(1, action.view(-1, 1)).squeeze(-1)
        loss = F.huber_loss(q_pred, q_trg)
        
        self.optim.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.online_q.parameters(), self.max_grad_norm)
        self.optim.step()
        
        return loss.item()