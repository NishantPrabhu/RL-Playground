
import os
import cv2
import time
import math
import wandb 
import torch
import random
import agents
import argparse
import numpy as np
import pandas as pd
import vizdoom as vzd
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from envs import env
from agents import dqn 
from datetime import datetime as dt
from utils import common, memory, cli_args
from gym.wrappers.monitoring.video_recorder import VideoRecorder


class Trainer:
    
    def __init__(self, args, seed=0):
        self.args = args 
        common.print_args(args)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.env, self.actions, self.action_names = env.VizdoomEnv(
            name=args.expt_name, 
            screen_format=args.screen_format,
            screen_res=args.screen_res
        )
        self.action_size = len(self.actions)
        
        self.agent = dqn.DQN(
            input_ch=args.frame_stack,
            n_actions=self.action_size,
            double_dqn=args.double_dqn,
            dueling_dqn=args.dueling_dqn,
            enc_hidden_ch=args.enc_hidden_ch,
            enc_fdim=args.enc_fdim,
            q_hidden_dim=args.q_hidden_dim,
            gamma=args.gamma,
            eps_max=args.eps_max,
            eps_min=args.eps_min,
            eps_decay_steps=args.eps_decay_steps,
            target_update_interval=args.target_update_interval,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm
        )
        self.memory = memory.ReplayMemory(
            mem_size=args.replay_mem_size, 
            obs_shape=(args.frame_height, args.frame_width, args.frame_stack)
        )
        self.generator = agents.networks.Generator(input_dim=(args.enc_fdim+self.action_size)).to(self.agent.device)
        self.gen_optim = optim.SGD(self.generator.parameters(), lr=args.learning_rate, momentum=0.9, nesterov=True)
    
        assert args.load is not None, f'Load a checkpoint!'        
        self.agent.load(args.load)

        self.out_dir = os.path.join('out', 'generative', 'vizdoom', args.expt_name, dt.now().strftime('%d-%m-%Y_%H-%M'))
        os.makedirs(self.out_dir, exist_ok=True)
        
        self.logger = common.Logger(self.out_dir)
        self.log_wandb = False
        if args.wandb:
            wandb.init(project='rl-playground', name='{}_{}'.format(args.expt_name, dt.now().strftime('%d-%m-%Y_%H-%M')))
            self.log_wandb = True
            
        if torch.cuda.is_available():
            self.logger.print('Found GPU device: {}'.format(torch.cuda.get_device_name(0)), mode='info')
        
        self.batch_size = self.args.batch_size
        self.best_val = float('inf')
        
    def _warp(self, frames):
        if frames[0].ndim == 2:
            frames = [np.expand_dims(f, -1) for f in frames]
        elif frames[0].ndim == 3 and frames[0].shape[-1] > 1:
            frames = [np.expand_dims(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), -1) for f in frames]
            
        obs = np.concatenate(frames, -1)
        obs = cv2.resize(obs, (self.args.frame_width, self.args.frame_height), interpolation=cv2.INTER_AREA)
        return obs
    
    @torch.no_grad()    
    def fill_memory(self):
        self.env.new_episode()
        for step in range(self.args.mem_init_steps):
            obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
            action = random.choice(np.arange(self.action_size))
            reward = self.env.make_action(self.actions[action], self.args.frame_skip)
            done = int(self.env.is_episode_finished())
            if not done:
                next_obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
            else:
                next_obs = np.zeros_like(obs, dtype=np.uint8)
                self.env.new_episode()
            
            self.memory.store(obs, action, reward, next_obs, done)
            common.pbar((step+1)/self.args.mem_init_steps, desc='Memory fill', status='')
        
    def train_epoch(self, epoch):
        avgmeter = common.AverageMeter()
        
        for step in range(self.args.steps_per_epoch):
            obs, action, _, next_obs, done = self.memory.get_batch(self.batch_size)
            obs = self.agent.preprocess_obs(obs)
            next_obs = self.agent.preprocess_obs(next_obs)
            action = torch.from_numpy(action).long().to(self.agent.device)
            done = torch.from_numpy(done).long().to(self.agent.device)
            
            with torch.no_grad():
                qvals, features, _ = self.agent.online_q(obs)
                action_onehot = F.one_hot(action, self.action_size)
                next_qvals, next_features, _ = self.agent.online_q(next_obs)
                next_action = next_qvals.argmax(-1)
                
            gen_inp = torch.cat([features, action_onehot], -1)    
            gen_out = self.generator(gen_inp)
            next_state_pred = torch.cat([obs[:, -(self.args.frame_stack-1):, :, :], gen_out], 1)            
            gen_qvals, gen_features, _ = self.agent.online_q(next_state_pred)
            
            # Losses
            q_loss = F.cross_entropy(gen_qvals, next_action)
            obs_recon_loss = F.huber_loss(gen_out.squeeze(1), next_obs[:, -1, :, :])
            fs_recon_loss = F.huber_loss(gen_features, next_features)
            loss = q_loss + obs_recon_loss + fs_recon_loss
            
            self.gen_optim.zero_grad()
            loss.backward()
            self.gen_optim.step()
            self.agent.optim.zero_grad()
            
            avgmeter.add({'q_loss': q_loss.item(), 'obs_recon_loss': obs_recon_loss.item(), 'fs_recon_loss': fs_recon_loss.item()})
            common.pbar((step+1)/self.args.steps_per_epoch, desc='Train Epoch {:3d}'.format(epoch), status=avgmeter.msg())
            
        self.logger.write('Train Epoch {:3d} {}'.format(epoch, avgmeter.msg()), mode='train')
        avg = avgmeter.get()
        if self.log_wandb:
            wandb.log({'train q_loss': avg['q_loss'], 'train obs_recon_loss': avg['obs_recon_loss'], 
                       'train fs_recon_loss': avg['fs_recon_loss'], 'epoch': epoch})
        
    @torch.no_grad()
    def evaluate(self, epoch):
        avgmeter = common.AverageMeter()
        
        for step in range(self.args.steps_per_epoch):
            obs, action, _, next_obs, done = self.memory.get_batch(self.batch_size)
            obs = self.agent.preprocess_obs(obs)
            next_obs = self.agent.preprocess_obs(next_obs)
            action = torch.from_numpy(action).long().to(self.agent.device)
            done = torch.from_numpy(done).long().to(self.agent.device)
            
            with torch.no_grad():
                qvals, features, _ = self.agent.online_q(obs)
                action_onehot = F.one_hot(action, self.action_size)
                next_qvals, next_features, _ = self.agent.online_q(next_obs)
                next_action = next_qvals.argmax(-1)
                
            gen_inp = torch.cat([features, action_onehot], -1)    
            gen_out = self.generator(gen_inp)
            next_state_pred = torch.cat([obs[:, -(self.args.frame_stack-1):, :, :], gen_out], 1)            
            gen_qvals, gen_features, _ = self.agent.online_q(next_state_pred)
            
            # Losses
            q_loss = F.cross_entropy(gen_qvals, next_action)
            obs_recon_loss = F.huber_loss(gen_out.squeeze(1), next_obs[:, -1, :, :])
            fs_recon_loss = F.huber_loss(gen_features, next_features)
            loss = q_loss + obs_recon_loss + fs_recon_loss
            
            avgmeter.add({'q_loss': q_loss.item(), 'obs_recon_loss': obs_recon_loss.item(), 'fs_recon_loss': fs_recon_loss.item(),
                          'total_loss': loss.item()})
            common.pbar((step+1)/self.args.steps_per_epoch, desc='Val Epoch   {:3d}'.format(epoch), status=avgmeter.msg())
            
        self.logger.write('Val Epoch  {:3d} {}'.format(epoch, avgmeter.msg()), mode='train')
        avg = avgmeter.get()
        if self.log_wandb:
            wandb.log({'val q_loss': avg['q_loss'], 'val obs_recon_loss': avg['obs_recon_loss'], 
                       'val fs_recon_loss': avg['fs_recon_loss'], 'epoch': epoch})
    
        if avg['total_loss'] < self.best_val:
            self.best_val = avg['total_loss']
            state = {'epoch': epoch, 'gen': self.generator.state_dict(), 'gen_optim': self.gen_optim.state_dict()}
            torch.save(state, os.path.join(self.out_dir, 'gen_state.pth.tar'))
    
    def run(self):
        self.fill_memory()
        
        for epoch in range(1, self.args.train_epochs):
            self.train_epoch(epoch)
            if epoch % self.args.eval_interval == 0:
                self.evaluate(epoch)
                
            if epoch % self.args.mem_refresh_interval == 0:
                self.fill_memory()
                
                
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser = cli_args.add_vizdoom_args(parser)
    args = parser.parse_args()
    
    trainer = Trainer(args)
    
    if args.task == 'train':
        trainer.run()