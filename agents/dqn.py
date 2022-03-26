
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
            self.optim = optim.RMSprop(self.online_q.parameters(), lr=learning_rate)
            if double_dqn:
                self.target_q = networks.QNetwork(input_ch, enc_hidden_ch, enc_fdim, q_hidden_dim, n_actions).to(self.device)
                self.target_q.load_state_dict(self.online_q.state_dict())
        else:
            self.online_q = networks.DuelQNetwork(input_ch, enc_hidden_ch, enc_fdim, q_hidden_dim, n_actions).to(self.device)
            self.optim = optim.RMSprop(self.online_q.parameters(), lr=learning_rate)
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
            
        q_pred = self.online_q(obs, replay=True)[0].gather(1, action.view(-1, 1)).squeeze(-1)
        loss = F.huber_loss(q_pred, q_trg)
        
        self.optim.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.online_q.parameters(), self.max_grad_norm)
        self.optim.step()
        
        return loss.item()
    
    
class ClusteringDQN:
    
    def __init__(
        self,
        input_ch: int,
        n_actions: int,
        n_clusters: int,
        double_dqn: bool = False,
        dueling_dqn: bool = False,
        enc_hidden_ch: int = 32,
        enc_fdim: int = 1024,
        q_hidden_dim: int = 1024,
        gamma: float = 0.99,
        eps_max: float = 1.0,
        eps_min: float = 0.01,
        eps_decay_steps: int = 500000,
        entropy_weight_max: float = 1.0,
        entropy_weight_min: float = 0.01,
        entropy_decay_steps: int = 500000,
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
        self.ew_max = entropy_weight_max
        self.ew_min = entropy_weight_min
        self.ent_weight = entropy_weight_max
        self.ew_decay_steps = (entropy_weight_max - entropy_weight_min) / entropy_decay_steps
        self.trg_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not dueling_dqn:
            self.online_q = networks.QNetwork(input_ch, enc_hidden_ch, enc_fdim, q_hidden_dim, n_actions).to(self.device)
            self.cls_head = networks.ClusteringHead(self.online_q.encoder.out_dim, q_hidden_dim, n_clusters).to(self.device)
            self.optim = optim.Adam(list(self.online_q.parameters())+list(self.cls_head.parameters()), lr=learning_rate)
            if double_dqn:
                self.target_q = networks.QNetwork(input_ch, enc_hidden_ch, enc_fdim, q_hidden_dim, n_actions).to(self.device)
                self.target_q.load_state_dict(self.online_q.state_dict())
        else:
            self.online_q = networks.DuelQNetwork(input_ch, enc_hidden_ch, enc_fdim, q_hidden_dim, n_actions).to(self.device)
            self.cls_head = networks.ClusteringHead(self.online_q.encoder.out_dim, q_hidden_dim, n_clusters).to(self.device)
            self.optim = optim.Adam(list(self.online_q.parameters())+list(self.cls_head.parameters()), lr=learning_rate)
            if double_dqn:
                self.target_q = networks.DuelQNetwork(input_ch, enc_hidden_ch, enc_fdim, q_hidden_dim, n_actions).to(self.device)
                self.target_q.load_state_dict(self.online_q.state_dict())
                
    def trainable(self, state=True):
        if state:
            self.online_q.train()
            self.target_q.train()
            self.cls_head.train()
        else:
            self.online_q.eval()
            self.target_q.eval()
            self.cls_head.eval()
            
    def save(self, out_dir):
        state = {'step': self.step, 'eps': self.eps, 'model_state': self.online_q.state_dict(), 
                 'cls_head_state': self.cls_head.state_dict()}
        torch.save(state, os.path.join(out_dir, 'ckpt.pth'))
        
    def load(self, ckpt_dir):
        file = os.path.join(ckpt_dir, 'ckpt.pth')
        if os.path.exists(file):
            state = torch.load(file, map_location=self.device)
            self.step = state['step'] 
            self.eps = state['eps']
            self.online_q.load_state_dict(state['model_state'])
            self.cls_head.load_state_dict(state['cls_head_state'])
            self.target_q.load_state_dict(state['model_state'])
        else:
            raise FileNotFoundError(f'Could not find model checkpoint at {ckpt_dir}')
            
    def update_epsilon(self):
        new_eps = self.eps_max - self.step * self.eps_decay_rate
        self.eps = max(new_eps, self.eps_min)
        
    def update_target_net(self):
        if self.step % self.trg_update_interval == 0:
            self.target_q.load_state_dict(self.online_q.state_dict())
            
    def update_entropy_weight(self):
        new_ew = self.ew_max - self.step * self.ew_decay_steps
        self.ent_weight = max(new_ew, self.eps_min)
            
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
        self.update_entropy_weight()
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
    
    def learn_clustering(self, batch):
        obs, _, _, next_obs, _ = batch 
        obs = self.preprocess_obs(obs)
        next_obs = self.preprocess_obs(next_obs)
        
        obs_fs = self.online_q(obs)[1]
        next_obs_fs = self.online_q(next_obs)[1]
        obs_cls = self.cls_head(obs_fs)
        next_obs_cls = self.cls_head(next_obs_fs)
        
        bs = obs_fs.size(0)
        mask = torch.ones((bs, bs)).long().fill_diagonal_(0).to(obs_fs.device)
        labels = torch.zeros(2 * bs).long().to(obs_fs.device)
        logits_ii = torch.mm(obs_cls, obs_cls.t())
        logits_ij = torch.mm(obs_cls, next_obs_cls.t())
        logits_ji = torch.mm(next_obs_cls, obs_cls.t())
        logits_jj = torch.mm(next_obs_cls, next_obs_cls.t())
        
        logits_ij_pos = logits_ij[torch.logical_not(mask)]
        logits_ji_pos = logits_ji[torch.logical_not(mask)]
        logits_ii_neg = logits_ii[mask].reshape(bs, -1)
        logits_ij_neg = logits_ij[mask].reshape(bs, -1)
        logits_ji_neg = logits_ji[mask].reshape(bs, -1)
        logits_jj_neg = logits_jj[mask].reshape(bs, -1)
        
        pos = torch.cat([logits_ij_pos, logits_ji_pos], 0).unsqueeze(-1)
        neg_i = torch.cat([logits_ii_neg, logits_ij_neg], 1)
        neg_j = torch.cat([logits_ji_neg, logits_jj_neg], 1)
        neg = torch.cat([neg_i, neg_j], 0)
        logits = torch.cat([pos, neg], 1)
        sim_loss = F.cross_entropy(logits, labels)

        p = torch.clamp(torch.mean(obs_cls, 0), min=1e-10)
        ent_loss = -(p * torch.log(p)).sum()
        
        loss = sim_loss - self.ent_weight * ent_loss
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        return sim_loss.item(), ent_loss.item()


class AttentionDQN:
    
    def __init__(
        self,
        input_ch: int,
        n_actions: int,
        input_res: tuple,
        enc_hidden_ch: int = 32,
        enc_fdim: int = 1024,
        q_hidden_dim: int = 1024,
        action_fdim: int = 128,
        attn_model_dim: int = 256,
        n_attn_heads: int = 8,
        n_attn_layers: int = 4,
        gamma: float = 0.99,
        eps_max: float = 1.0,
        eps_min: float = 0.01,
        eps_decay_steps: int = 500000,
        target_update_interval: int = 10000,
        learning_rate: float = 0.0001,
        max_grad_norm: float = 1.0,
        encoder_init_type: str = 'dueling',
        encoder_init_ckpt_dir: str = None
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
        
        self.online_q = networks.AttentionQNetwork(
            input_ch, enc_hidden_ch, enc_fdim, n_actions, input_res, 
            n_attn_heads, n_attn_layers, attn_model_dim, action_fdim
        ).to(self.device)
        self.optim = optim.Adam(self.online_q.parameters(), lr=learning_rate)
        
        if encoder_init_type == 'normal':
            self.enc_buffer = networks.QNetwork(input_ch, enc_hidden_ch, enc_fdim, q_hidden_dim, n_actions).to(self.device)
        elif encoder_init_type == 'dueling':
            self.enc_buffer = networks.DuelQNetwork(input_ch, enc_hidden_ch, enc_fdim, q_hidden_dim, n_actions).to(self.device)
        
        if encoder_init_ckpt_dir is not None:
            self.init_encoder(encoder_init_ckpt_dir)
        
        for p in self.enc_buffer.parameters():
            p.requires_grad = False
                    
    def save(self, out_dir):
        state = {'step': self.step, 'eps': self.eps, 'model_state': self.online_q.state_dict()}
        torch.save(state, os.path.join(out_dir, 'ckpt.pth'))
        
    def trainable(self, state=True):
        if state:
            self.online_q.train()
        else:
            self.online_q.eval()
        
    def load(self, ckpt_dir):
        file = os.path.join(ckpt_dir, 'ckpt.pth')
        if os.path.exists(file):
            state = torch.load(file, map_location=self.device)
            self.step = state['step']
            self.eps = state['eps']
            self.online_q.load_state_dict(state['model_state'])
        else:
            raise FileNotFoundError(f'Could not find model checkpoint at {ckpt_dir}')
        
    def init_encoder(self, enc_init_ckpt_dir):
        file = os.path.join(enc_init_ckpt_dir, 'ckpt.pth')
        if os.path.exists(file):
            state = torch.load(file, map_location=self.device)
            self.enc_buffer.load_state_dict(state['model_state'])
            self.online_q.load_encoder(self.enc_buffer)
        else:
            raise FileNotFoundError(f'Could not find model checkpoint at {enc_init_ckpt_dir}')
            
    def preprocess_obs(self, obs):
        if np.asarray(obs).ndim == 3:
            obs = np.asarray(obs).transpose((2, 0, 1))
            obs = torch.from_numpy(obs).to(self.device) / 255.0
            obs = obs.unsqueeze(0)
        elif np.asarray(obs).ndim == 4:
            obs = np.asarray(obs).transpose((0, 3, 1, 2)) 
            obs = torch.from_numpy(obs).to(self.device) / 255.0
        return obs
    
    def loss_fn(self, output, target):
        output = F.softmax(output, dim=1)
        target = target.argmax(dim=1)
        loss = F.cross_entropy(output, target)
        acc = torch.eq(output.argmax(1), target).sum().item() / target.size(0)
        return loss, acc
            
    def select_action(self, obs):
        obs = self.preprocess_obs(obs)
        
        if random.uniform(0, 1) < self.eps:
            action = random.randint(0, self.n_actions-1)
        else:
            with torch.no_grad():
                action = self.enc_buffer(obs)[0].argmax(-1).item()
        
        self.step += 1
        return action
    
    def select_agent_action(self, obs):
        obs = self.preprocess_obs(obs)
        with torch.no_grad():
            action = self.enc_buffer(obs)[0].argmax(-1).item()
        return action
    
    def learn_from_memory(self, batch):
        obs, action, _, _, _ = batch 
        obs = self.preprocess_obs(obs)
        action = torch.from_numpy(action).long().to(obs.device)
        with torch.no_grad():
            trg_q, _, _ = self.enc_buffer(obs) 
        
        pred_q, _, _, attn_probs = self.online_q(obs)
        loss, acc = self.loss_fn(pred_q, trg_q)
        
        self.optim.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.online_q.parameters(), self.max_grad_norm)
        self.optim.step()
        return loss.item(), acc, attn_probs
    
    
class VectorizedActionDQN:
    
    def __init__(
        self,
        input_ch: int,
        n_actions: int,
        double_dqn: bool = False,
        enc_hidden_ch: int = 32,
        enc_fdim: int = 1024,
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
        
        self.online_q = networks.VectorizedActionQNetwork(input_ch, enc_hidden_ch, enc_fdim, n_actions).to(self.device)
        self.optim = optim.Adam(self.online_q.parameters(), lr=learning_rate)
        if double_dqn:
            self.target_q = networks.VectorizedActionQNetwork(input_ch, enc_hidden_ch, enc_fdim, n_actions).to(self.device)
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