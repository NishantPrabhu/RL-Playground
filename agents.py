
import os
import math
import torch 
import random
import networks 
import numpy as np
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 


class ReplayMemory:
    """ Observations will be stored and retrieved in NHWC format """
    
    def __init__(self, mem_size, obs_shape):
        self.ptr = 0
        self.filled = 0
        self.mem_size = mem_size
        
        self.obses = np.zeros((self.mem_size, *obs_shape), dtype=np.uint8)
        self.next_obses = np.zeros((self.mem_size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((self.mem_size,), dtype=np.uint8)
        self.rewards = np.zeros((self.mem_size), dtype=np.float32)
        self.dones = np.zeros((self.mem_size,), dtype=np.uint8)
        
    def store(self, obs, action, reward, next_obs, done):
        self.obses[self.ptr] = np.array(obs)
        self.next_obses[self.ptr] = np.array(next_obs)
        self.actions[self.ptr] = action 
        self.rewards[self.ptr] = reward 
        self.dones[self.ptr] = int(done) 
        
        self.ptr = (self.ptr + 1) % self.mem_size 
        self.filled = min(self.filled + 1, self.mem_size)
        
    def get_batch(self, batch_size):
        assert self.filled > batch_size, 'Not enough samples in memory yet'
        idx = np.random.choice(np.arange(self.filled), size=batch_size, replace=False)
        
        obs = self.obses[idx]
        next_obs = self.next_obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        done = self.dones[idx]
        return (obs, action, reward, next_obs, done)


class ViTDQN:
    
    def __init__(
        self,
        input_shape: tuple,
        n_actions: int,
        double_dqn: bool = False,
        duel_dqn: bool = False,
        n_layers: int = 4,
        n_heads: int = 4,
        model_dim: int = 256,
        patch_size: int = 4,
        n_embeds: int = 1024,
        add_action_embeds: bool = True,
        gamma: float = 0.99,
        eps_max: float = 1.0,
        eps_min: float = 0.01,
        eps_decay_steps: int = 500_000,
        trg_update_interval: int = 10_000,
        base_lr: float = 0.001,
        min_lr: float = 1e-10,
        lr_decay_steps: int = 10_000_000,
        lr_decay_factor: float = 0.5,
        weight_decay: float = 1e-05,
        max_grad_norm: float = 1.0,
        replay_mem_size: int = 500_000,
        replay_batch_size: int = 32
    ):
        self.step = 0
        self.gamma = gamma 
        self.n_actions = n_actions
        self.eps_min = eps_min
        self.eps_max = eps_max 
        self.eps = eps_max 
        self.eps_decay_rate = (eps_max - eps_min) / eps_decay_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.lr_decay_steps = lr_decay_steps
        self.lr_decay_factor = lr_decay_factor
        self.trg_update_interval = trg_update_interval
        self.max_grad_norm = max_grad_norm 
        self.double_dqn = double_dqn
        self.replay_bs = replay_batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.encoder = networks.Encoder(
            input_shape=input_shape,
            n_layers=n_layers,
            n_heads=n_heads,
            model_dim=model_dim,
            patch_size=patch_size,
            n_embeds=n_embeds,
            add_action_embeds=add_action_embeds,
            n_actions=n_actions
        )
        if not duel_dqn:
            self.online_q = networks.DuelQNetwork(model_dim, model_dim, n_actions)
            if double_dqn:
                self.target_q = networks.DuelQNetwork(model_dim, model_dim, n_actions)
                self.target_q.load_state_dict(self.online_q.state_dict())
        else:
            self.online_q = networks.QNetwork(model_dim, model_dim, n_actions)
            if double_dqn:
                self.target_q = networks.QNetwork(model_dim, model_dim, n_actions)
                self.target_q.load_state_dict(self.online_q.state_dict())
                
        self.trainable_params = list(self.encoder.parameters()) + list(self.online_q.parameters())
        self.optim = optim.Adam(self.trainable_params, lr=base_lr, weight_decay=weight_decay)
        self.memory = ReplayMemory(replay_mem_size, input_shape)
        
    def trainable(self, state=True):
        if state:
            self.encoder.train()
            self.online_q.train()
            self.target_q.train()
        else:
            self.encoder.eval()
            self.online_q.eval()
            self.target_q.eval()
            
    def save(self, out_dir):
        state = {
            'step': self.step,
            'encoder': self.encoder.state_dict(), 
            'qnet': self.online_q.state_dict(),
            'optim': self.optim.state_dict()
        }
        torch.save(state, os.path.join(self.out_dir, 'checkpoint.pth.tar'))
    
    def load(self, ckpt_dir):
        fp = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
        if os.path.exists(fp):
            state = torch.load(fp, map_location=self.device)
            self.encoder.load_state_dict(state['encoder'])
            self.online_q.load_state_dict(state['qnet'])
            self.target_q.load_state_dict(state['qnet'])
            self.optim.load_state_dict(state['optim'])
            self.step = state['step'] 
        else:
            raise FileNotFoundError(f'Could not find checkpoint.pth.tar at {ckpt_dir}')
        
    def update_epsilon(self):
        new_eps = self.eps_max - self.step * self.eps_decay_rate
        self.eps = max(new_eps, self.eps_min)
        
    def update_target_q(self):
        if self.step % self.trg_update_interval == 0:
            self.target_q.load_state_dict(self.online_q.state_dict())
            
    def decay_lr(self):
        factor = math.pow(1.0 - self.step/self.lr_decay_steps, self.lr_decay_factor)
        new_lr = max(self.base_lr * factor, self.min_lr)
        for group in self.optim.param_groups:
            group['lr'] = new_lr
            
    def preprocess_obs(self, frames):
        assert np.asarray(frames).ndim == 4, f'Expected input with dim 4, got shape {np.asarray(frames).shape}'
        obs = np.asarray(frames).transpose((0, 3, 1, 2))
        obs = torch.from_numpy(obs).float().to(self.device) / 255.0
        return obs 
    
    def sample_action(self, frames):
        obs = self.preprocess_obs(frames)
        if random.uniform(0, 1) < self.eps:
            action = random.randint(0, self.n_actions-1)
        else:
            with torch.no_grad():
                action = self.online_q(obs).argmax(-1).item()
        
        self.step += 1
        self.update_epsilon()
        if self.double_dqn:
            self.update_target_q()
        return action 
    
    def select_action(self, frames):
        obs = self.preprocess_obs(frames)
        with torch.no_grad():
            action = self.online_q(obs).argmax(-1).item()
        return action
    
    def experience_replay(self):
        obs, action, reward, next_obs, done = self.memory.get_batch(self.replay_bs) 
        obs = self.preprocess_obs(obs)
        next_obs = self.preprocess_obs(next_obs)
        action = torch.from_numpy(action).long().to(self.device)
        reward = torch.from_numpy(reward).float().to(self.device)
        done = torch.from_numpy(done).long().to(self.device)
        
        with torch.no_grad():
            next_action = self.online_q(next_obs).argmax(-1)
            if self.double_dqn:
                next_qval = self.target_q(next_obs).gather(1, next_action.view(-1, 1)).squeeze(-1)
            else:
                next_qval = self.online_q(next_obs).gather(1, next_action.view(-1, 1)).squeeze(-1)
            q_trg = reward + (1-done) * self.gamma * next_qval 
            
        q_pred = self.online(obs).gather(1, action.view(-1, 1)).squeeze(-1)
        loss = F.huber_loss(q_pred, q_trg)
        
        self.optim.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.trainable_params, self.max_grad_norm)
        self.optim.step()
        
        self.decay_lr()
        return loss.item() 