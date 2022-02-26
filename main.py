
import os
import math
import wandb 
import torch
import utils
import agents
import argparse 
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from envs import env
from datetime import datetime as dt 
from gym.wrappers.monitoring.video_recorder import VideoRecorder


class Trainer:
    
    def __init__(self, args, seed=0):
        self.args = args 
        utils.print_args(args)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        self.env = env.AtariEnv(
            name=args.expt_name, 
            frame_res=(args.frame_height, args.frame_width), 
            frame_skip=args.frame_skip, 
            stack_frames=args.frame_stack, 
            reset_noops=args.reset_noops, 
            episodic_life=args.episodic_life,
            clip_rewards=args.clip_rewards
        )
        self.agent = agents.ViTDQN(
            input_shape=(args.frame_height, args.frame_width, args.frame_stack), 
            n_actions=self.env.action_space.n,
            encoder_type=args.encoder_type,
            double_dqn=args.double_dqn,
            duel_dqn=args.duel_dqn,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            model_dim=args.model_dim,
            patch_size=args.patch_size,
            n_embeds=args.n_embeds,
            add_action_embeds=args.add_action_embeds,
            gamma=args.gamma,
            eps_max=args.eps_max,
            eps_min=args.eps_min,
            eps_decay_steps=args.eps_decay_steps,
            trg_update_interval=args.trg_update_interval,
            base_lr=args.base_lr,
            min_lr=args.min_lr,
            lr_decay_steps=args.lr_decay_steps,
            lr_decay_factor=args.lr_decay_factor,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            replay_mem_size=args.replay_mem_size,
            replay_batch_size=args.replay_batch_size
        )
        
        if args.load is not None:
            self.agent.load(args.load)
            self.out_dir = args.load
        else:        
            self.out_dir = os.path.join('out', args.expt_name, dt.now().strftime('%d-%m-%Y_%H-%M'))
            os.makedirs(self.out_dir, exist_ok=True)
            os.makedirs(os.path.join(self.out_dir, 'videos'), exist_ok=True)
        
        self.best_train, self.best_val = 0, 0
        self.logger = utils.Logger(self.out_dir)
        self.log_wandb = False 
        if args.wandb:
            wandb.init(project='vit-rl', name='{}_{}'.format(args.expt_name, dt.now().strftime('%d-%m-%Y_%H-%M')))
            self.log_wandb = True
            
        if torch.cuda.is_available():
            self.logger.print('Found GPU device: {}'.format(torch.cuda.get_device_name(0)), mode='info')
                
    def init_memory(self):
        obs = self.env.reset()
        for step in range(self.args.mem_init_steps):
            action = self.env.action_space.sample()
            next_obs, reward, done, _ = self.env.step(action)
            if reward > self.best_train:
                self.best_train = reward
            
            self.agent.memory.store(obs, action, reward, next_obs, done)
            obs = next_obs if not done else self.env.reset()
            utils.pbar((step+1)/self.args.mem_init_steps, desc='Progress', status='')
            
        total_loss = 0
        random_steps = self.args.mem_init_steps // self.args.replay_batch_size
        for step in range(random_steps):
            loss = self.agent.experience_replay()
            total_loss += loss
            utils.pbar((step+1)/random_steps, desc="Learning", status='')

        avg_loss = total_loss / random_steps
        self.logger.record('Memory init: [loss] {:.4f} [train_best] {}'.format(avg_loss, self.best_train), mode='train')
        
    def train_episode(self, episode):
        self.agent.trainable(True)
        meter = utils.AverageMeter()
        episode_done = False 
        total_reward = 0
        step = 0
        
        obs = self.env.reset()
        while not episode_done:
            action = self.agent.sample_action(obs)
            next_obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            step += 1
            
            self.agent.memory.store(obs, action, reward, next_obs, done)
            if done:
                episode_done = True
            else:
                obs = next_obs
                
            if step % self.args.exp_replay_interval == 0:
                loss = self.agent.experience_replay()
                meter.add({'loss': loss})
                
        if total_reward > self.best_train:
            self.best_train = round(total_reward)
            
        if episode % self.args.log_interval == 0:
            self.logger.record("Episode: {:<7d} [reward] {:<5d} {} [train_best] {:<5d} [val_best] {:<5d} [eps] {:.4f}".format(
                episode, round(total_reward), meter.msg(), round(self.best_train), round(self.best_val), self.agent.eps),
                mode='train'
            )
        if self.log_wandb:
            wandb.log({'episode': episode, 'reward': total_reward, 'epsilon': self.agent.eps, **meter.get()})

    @torch.no_grad()
    def evaluate(self, episode):
        self.agent.trainable(False)
        meter = utils.AverageMeter()
            
        for _ in range(self.args.eval_episodes):
            episode_done = False 
            total_reward = 0
            
            obs = self.env.reset()
            while not episode_done:
                action = self.agent.select_action(obs)
                next_obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                
                if done:
                    episode_done = True
                else:
                    obs = next_obs
            meter.add({'reward': total_reward})
                
        avg_reward = meter.get()['reward']
        if avg_reward > self.best_val:
            self.best_val = round(avg_reward)
            self.agent.save(self.out_dir)
              
        self.logger.record("Episode: {:<7d} [reward] {:<5d} [loss] N/A    [train_best] {:<5d} [val_best] {:<5d}".format(
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
        for i in range(5):
            rec = VideoRecorder(
                self.env, 
                path=os.path.join(self.out_dir, "videos", f"attempt_{i}.mp4"), 
                enabled=True
            )
            self.agent.trainable(False)
            episode_done = False
            total_reward = 0
            
            obs = self.env.reset()
            rec.capture_frame()
            while not episode_done:
                action = self.agent.select_action(obs)
                next_obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                rec.capture_frame()

                if done:
                    episode_done = True
                else:
                    obs = next_obs
            
            rec.close()
            rec.enabled = False
            self.env.close()
            self.logger.print("Attempt {:<2d} [reward] {}".format(i, total_reward), mode='val')

    
            
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser = utils.parse_args(parser)
    args = parser.parse_args()
    
    trainer = Trainer(args)
    
    if args.task == 'train':
        trainer.run()
    
    elif args.task == 'record':
        assert args.load is not None, 'Model checkpoint required for recording video'
        trainer.create_video()