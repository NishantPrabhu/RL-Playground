
import os
import cv2
import time
import wandb 
import torch
import random
import argparse 
import numpy as np
import vizdoom as vzd

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
        
        self.env, self.actions = env.VizdoomEnv(
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
        
        if args.load is not None:
            self.agent.load(args.load)
            self.out_dir = args.load
        else:        
            self.out_dir = os.path.join('out', 'vizdoom', args.expt_name, dt.now().strftime('%d-%m-%Y_%H-%M'))
            os.makedirs(self.out_dir, exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, 'videos'), exist_ok=True)
        
        self.logger = common.Logger(self.out_dir)
        self.log_wandb = False
        if args.wandb:
            wandb.init(project='rl-playground', name='{}_{}'.format(args.expt_name, dt.now().strftime('%d-%m-%Y_%H-%M')))
            self.log_wandb = True
            
        if torch.cuda.is_available():
            self.logger.print('Found GPU device: {}'.format(torch.cuda.get_device_name(0)), mode='info')
        
        self.batch_size = self.args.batch_size
        self.best_train, self.best_val = 0, 0
        
    def _warp(self, frames):
        if frames[0].ndim == 2:
            frames = [np.expand_dims(f, -1) for f in frames]
        elif frames[0].ndim == 3 and frames[0].shape[-1] > 1:
            frames = [np.expand_dims(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), -1) for f in frames]
            
        obs = np.concatenate(frames, -1)
        obs = cv2.resize(obs, (self.args.frame_width, self.args.frame_height), interpolation=cv2.INTER_AREA)
        return obs
        
    def init_memory(self):
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
            common.pbar((step+1)/self.args.mem_init_steps, desc='Progress', status='')
            
        total_loss = 0
        random_steps = self.args.mem_init_steps // self.batch_size
        for step in range(random_steps):
            batch = self.memory.get_batch(self.batch_size)
            loss = self.agent.learn_from_memory(batch)
            total_loss += loss
            common.pbar((step+1)/random_steps, desc="Learning", status='')
        print()
        avg_loss = total_loss / random_steps
        self.logger.record('QL: {:.4f} | BEST_T: {}'.format(avg_loss, self.best_train), mode='train')
        
    def train_episode(self, episode):
        self.agent.trainable(True)
        meter = common.AverageMeter()
        episode_done = False 
        step = 0
        
        self.env.new_episode()
        while not episode_done:
            obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
            action = self.agent.select_action(obs)
            reward = self.env.make_action(self.actions[action], self.args.frame_skip)
            done = int(self.env.is_episode_finished())
            if not done:
                next_obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
            else:
                next_obs = np.zeros_like(obs, dtype=np.uint8)
                episode_done = True                
            
            self.memory.store(obs, action, reward, next_obs, done)
            step += 1
            
            if step % self.args.mem_replay_interval == 0:
                batch = self.memory.get_batch(self.batch_size)
                loss = self.agent.learn_from_memory(batch)
                meter.add({'QL': loss})
                
        total_reward = self.env.get_total_reward()
        if total_reward > self.best_train:
            self.best_train = round(total_reward)
            
        if episode % self.args.log_interval == 0:
            self.logger.record("E: {:7d} | R: {:4d} |{} BEST_T: {:3d} | BEST_V: {:3d} | EPS: {:.2f}".format(
                episode, round(total_reward), meter.msg(), round(self.best_train), round(self.best_val), self.agent.eps),
                mode='train'
            )
        if self.log_wandb:
            wandb.log({'episode': episode, 'reward': total_reward, **meter.get()})

    @torch.no_grad()
    def evaluate(self, episode):
        self.agent.trainable(False)
        meter = common.AverageMeter()
            
        for _ in range(self.args.eval_episodes):
            episode_done = False
            total_reward = 0
            
            self.env.new_episode()
            while not episode_done:
                obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
                action = self.agent.select_agent_action(obs)
                reward = self.env.make_action(self.actions[action], self.args.frame_skip)
                done = int(self.env.is_episode_finished())
                if not done:
                    next_obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
                else:
                    next_obs = np.zeros_like(obs, dtype=np.uint8)
                    episode_done = True 

            total_reward = self.env.get_total_reward()
            meter.add({'R': total_reward})
                
        avg_reward = meter.get()['R']
        if avg_reward > self.best_val:
            self.best_val = round(avg_reward)
            self.agent.save(self.out_dir)
              
        self.logger.record("E: {:7d} | R: {:4d} | BEST_T: {:3d} | BEST_V: {:3d}".format(
            episode, round(avg_reward), round(self.best_train), round(self.best_val)),
            mode='val'
        )
        if self.log_wandb:
            wandb.log({'episode': episode, 'val_reward': avg_reward})
    
    def run(self):
        self.logger.print('Initializing memory', mode='info')
        self.init_memory()
        print()
        self.logger.print('Beginning training', mode='info')
        
        for episode in range(1, self.args.train_episodes+1):
            self.train_episode(episode)
            if episode % self.args.eval_interval == 0:
                self.evaluate(episode)
                
        self.logger.print('Completed training', mode='info')
        
    @torch.no_grad()
    def create_video(self):
        self.env.close()
        self.env.set_window_visible(True)
        self.env.set_mode(vzd.Mode.ASYNC_PLAYER)
        self.env.init()
        self.agent.trainable(False)
        
        for i in range(self.args.spectator_episodes):
            self.env.new_episode(os.path.join(self.out_dir, 'videos', f'attempt_{i}.lmp'))
            
            while not self.env.is_episode_finished():
                obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
                action = self.agent.select_agent_action(obs)
                reward = self.env.set_action(self.actions[action])
                for _ in range(self.args.frame_skip):
                    self.env.advance_action()
                    
            time.sleep(1.0)
            total_reward = self.env.get_total_reward()
            self.logger.print('E: {:7d} | R: {:4d}'.format(i+1, round(total_reward)))
            
        self.env.close()
        
        
class ClusteringTrainer:
    
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
        
        self.env, self.actions = env.VizdoomEnv(
            name=args.expt_name, 
            screen_format=args.screen_format,
            screen_res=args.screen_res
        )
        self.action_size = len(self.actions)
        
        self.agent = dqn.ClusteringDQN(
            input_ch=args.frame_stack,
            n_actions=self.action_size,
            n_clusters=args.n_clusters,
            double_dqn=args.double_dqn,
            dueling_dqn=args.dueling_dqn,
            enc_hidden_ch=args.enc_hidden_ch,
            enc_fdim=args.enc_fdim,
            q_hidden_dim=args.q_hidden_dim,
            gamma=args.gamma,
            eps_max=args.eps_max,
            eps_min=args.eps_min,
            eps_decay_steps=args.eps_decay_steps,
            entropy_weight_max=args.entropy_weight_max,
            entropy_weight_min=args.entropy_weight_min,
            entropy_decay_steps=args.entropy_decay_steps,
            target_update_interval=args.target_update_interval,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm
        )
        self.memory = memory.ReplayMemory(
            mem_size=args.replay_mem_size, 
            obs_shape=(args.frame_height, args.frame_width, args.frame_stack)
        )
        
        if args.load is not None:
            self.agent.load(args.load)
            self.out_dir = args.load
        else:        
            self.out_dir = os.path.join('out', 'vizdoom', args.expt_name, dt.now().strftime('%d-%m-%Y_%H-%M'))
            os.makedirs(self.out_dir, exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, 'videos'), exist_ok=True)
        
        self.logger = common.Logger(self.out_dir)
        self.log_wandb = False
        if args.wandb:
            wandb.init(project='rl-playground', name='{}_{}'.format(args.expt_name, dt.now().strftime('%d-%m-%Y_%H-%M')))
            self.log_wandb = True
            
        if torch.cuda.is_available():
            self.logger.print('Found GPU device: {}'.format(torch.cuda.get_device_name(0)), mode='info')
        
        self.batch_size = self.args.batch_size
        self.cls_batch_size = self.args.cls_batch_size
        self.best_train, self.best_val = 0, 0
        
    def _warp(self, frames):
        if frames[0].ndim == 2:
            frames = [np.expand_dims(f, -1) for f in frames]
        elif frames[0].ndim == 3 and frames[0].shape[-1] > 1:
            frames = [np.expand_dims(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), -1) for f in frames]
            
        obs = np.concatenate(frames, -1)
        obs = cv2.resize(obs, (self.args.frame_width, self.args.frame_height), interpolation=cv2.INTER_AREA)
        return obs
        
    def init_memory(self):
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
            common.pbar((step+1)/self.args.mem_init_steps, desc='Progress', status='')
            
        total_loss = 0
        random_steps = self.args.mem_init_steps // self.batch_size
        for step in range(random_steps):
            batch = self.memory.get_batch(self.batch_size)
            loss = self.agent.learn_from_memory(batch)
            total_loss += loss
            common.pbar((step+1)/random_steps, desc="Learning", status='')
        print()
        avg_loss = total_loss / random_steps
        self.logger.record('QL: {:.4f} | BEST_T: {}'.format(avg_loss, self.best_train), mode='train')
        
    def train_episode(self, episode):
        self.agent.trainable(True)
        meter = common.AverageMeter()
        episode_done = False 
        step = 0
        
        self.env.new_episode()
        while not episode_done:
            obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
            action = self.agent.select_action(obs)
            reward = self.env.make_action(self.actions[action], self.args.frame_skip)
            done = int(self.env.is_episode_finished())
            if not done:
                next_obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
            else:
                next_obs = np.zeros_like(obs, dtype=np.uint8)
                episode_done = True                
            
            self.memory.store(obs, action, reward, next_obs, done)
            step += 1
            
            if step % self.args.mem_replay_interval == 0:
                batch = self.memory.get_batch(self.batch_size)
                loss = self.agent.learn_from_memory(batch)
                meter.add({'QL': loss})
            
            if step % self.args.cls_learning_interval == 0:
                batch = self.memory.get_batch(self.cls_batch_size)
                sim_loss, ent_loss = self.agent.learn_clustering(batch)
                meter.add({'SL': sim_loss, 'EL': ent_loss})
                
        total_reward = self.env.get_total_reward()
        if total_reward > self.best_train:
            self.best_train = round(total_reward)
            
        if episode % self.args.log_interval == 0:
            self.logger.record("E: {:7d} | R: {:4d} |{} BEST_T: {:3d} | BEST_V: {:3d} | EPS: {:.2f}".format(
                episode, round(total_reward), meter.msg(), round(self.best_train), round(self.best_val), self.agent.eps),
                mode='train'
            )
        if self.log_wandb:
            wandb.log({'episode': episode, 'reward': total_reward, **meter.get()})

    @torch.no_grad()
    def evaluate(self, episode):
        self.agent.trainable(False)
        meter = common.AverageMeter()
            
        for _ in range(self.args.eval_episodes):
            episode_done = False
            total_reward = 0
            
            self.env.new_episode()
            while not episode_done:
                obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
                action = self.agent.select_agent_action(obs)
                reward = self.env.make_action(self.actions[action], self.args.frame_skip)
                done = int(self.env.is_episode_finished())
                if not done:
                    next_obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
                else:
                    next_obs = np.zeros_like(obs, dtype=np.uint8)
                    episode_done = True 

            total_reward = self.env.get_total_reward()
            meter.add({'R': total_reward})
                
        avg_reward = meter.get()['R']
        if avg_reward > self.best_val:
            self.best_val = round(avg_reward)
            self.agent.save(self.out_dir)
              
        self.logger.record("E: {:7d} | R: {:4d} | BEST_T: {:3d} | BEST_V: {:3d}".format(
            episode, round(avg_reward), round(self.best_train), round(self.best_val)),
            mode='val'
        )
        if self.log_wandb:
            wandb.log({'episode': episode, 'val_reward': avg_reward})
    
    def run(self):
        self.logger.print('Initializing memory', mode='info')
        self.init_memory()
        print()
        self.logger.print('Beginning training', mode='info')
        
        for episode in range(1, self.args.train_episodes+1):
            self.train_episode(episode)
            if episode % self.args.eval_interval == 0:
                self.evaluate(episode)
                
        self.logger.print('Completed training', mode='info')
        
    @torch.no_grad()
    def create_video(self):
        self.env.close()
        self.env.set_window_visible(True)
        self.env.set_mode(vzd.Mode.ASYNC_PLAYER)
        self.env.init()
        self.agent.trainable(False)
        
        for i in range(self.args.spectator_episodes):
            self.env.new_episode(os.path.join(self.out_dir, 'videos', f'attempt_{i}.lmp'))
            
            while not self.env.is_episode_finished():
                obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
                action = self.agent.select_agent_action(obs)
                reward = self.env.set_action(self.actions[action])
                for _ in range(self.args.frame_skip):
                    self.env.advance_action()
                    
            time.sleep(1.0)
            total_reward = self.env.get_total_reward()
            self.logger.print('E: {:7d} | R: {:4d}'.format(i+1, round(total_reward)))
            
        self.env.close()

            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser = cli_args.add_vizdoom_args(parser)
    args = parser.parse_args()
    
    trainer = ClusteringTrainer(args)
    
    if args.task == 'train':
        trainer.run()
    
    elif args.task == 'record':
        assert args.load is not None, 'Model checkpoint required for recording video'
        trainer.create_video()