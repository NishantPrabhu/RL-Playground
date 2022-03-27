
import os
import cv2
import time
import math
import wandb 
import torch
import faiss
import random
import argparse
import numpy as np
import pandas as pd
import vizdoom as vzd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from tqdm import tqdm
from envs import env
from agents import dqn 
from sklearn.decomposition import PCA
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
        self.n_actions = len(self.actions[0])
        
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
        
    @torch.no_grad()
    def automap_viz(self):
        self.env.close()
        self.env.set_mode(vzd.Mode.ASYNC_PLAYER)
        self.env.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        self.env.set_screen_format(vzd.ScreenFormat.RGB24)
        self.env.set_automap_buffer_enabled(True)
        self.env.set_automap_mode(vzd.AutomapMode.OBJECTS_WITH_SIZE)
        self.env.add_available_game_variable(vzd.GameVariable.POSITION_X)
        self.env.add_available_game_variable(vzd.GameVariable.POSITION_Y)
        self.env.add_available_game_variable(vzd.GameVariable.POSITION_Z)
        self.env.set_window_visible(False)
        self.env.add_game_args("+am_followplayer 1")
        self.env.add_game_args("+viz_am_scale 10")
        self.env.add_game_args("+viz_am_center 1")
        self.env.add_game_args("+am_backcolor 000000")
        self.env.add_game_args("+am_yourcolor ffff00")
        self.env.add_game_args("+am_thingcolor_monster ff0000")
        self.env.add_game_args("+am_thingcolor_item 00ff00")
        self.env.init()
        
        self.agent.trainable(False)
        meter = common.AverageMeter()
        fig, axarr = plt.subplots(1, 2, figsize=(10, 4))
        total_rewards = []
        images = []
            
        for j in range(self.args.eval_episodes):
            episode_done = False
            total_reward = 0
            
            self.env.new_episode()
            while not episode_done:
                obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
                action = self.agent.select_agent_action(obs)
                reward = self.env.set_action(self.actions[action])
                
                for _ in range(self.args.frame_skip):
                    self.env.advance_action()
                    if not int(self.env.is_episode_finished()):
                        im1 = axarr[0].imshow(self.env.get_state().screen_buffer, cmap='gray')
                        axarr[0].axis('off')
                        im2 = axarr[1].imshow(self.env.get_state().automap_buffer, cmap='gray')
                        axarr[1].axis('off')
                        images.append([im1, im2])
                        
                done = int(self.env.is_episode_finished())
                    
                if not done:
                    next_obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
                else:
                    next_obs = np.zeros_like(obs, dtype=np.uint8)
                    episode_done = True

            total_rewards.append(self.env.get_total_reward())
            print(f"Episode {j+1}: Total reward = {self.env.get_total_reward()}")
            
        anim = animation.ArtistAnimation(fig, images, interval=10000, blit=True)
        anim.save(os.path.join(self.out_dir, 'videos', 'automap_viz.mp4'.format(np.mean(total_rewards))), fps=24)            
        
    def transition_probs(self, labels):
        visit_probs = {}
        n_txn = 0

        for i in range(0, len(labels)-1):
            s, s_ = labels[i], labels[i+1]
            name = f'{s}_{s_}'
            n_txn += 1
            
            if name not in visit_probs:
                visit_probs[name] = 1
            else:
                visit_probs[name] += 1
            
        visit_probs = {k: v/n_txn for k, v in visit_probs.items()}
        return visit_probs        
        
    @torch.no_grad()
    def convert_to_mdp(self):
        os.makedirs(os.path.join(self.out_dir, 'mdp_viz'), exist_ok=True)
        self.agent.trainable(False)
        features_buffer = []
        actions_buffer = []

        for _ in tqdm(range(100)):        
            episode_done = False 
            total_reward = 0
            step = 0
            
            self.env.new_episode()
            while not episode_done:
                obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
                action = self.agent.select_action(obs)
                reward = self.env.make_action(self.actions[action], self.args.frame_skip)
                done = int(self.env.is_episode_finished())
                step += 1
                
                state_fs, _ = self.agent.online_q.encoder(self.agent.preprocess_obs(obs))
                state_fs = state_fs.detach().cpu().numpy()
                features_buffer.append(state_fs)
                actions_buffer.append(action)
                
                if not done:
                    next_obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
                else:
                    next_obs = np.zeros_like(obs, dtype=np.uint8)
                    episode_done = True
                    
        features = np.concatenate(features_buffer, 0).astype(np.float32)
        kmeans = faiss.Kmeans(
            features.shape[1], self.n_actions, niter=20, verbose=False, gpu=torch.cuda.is_available()
        )
        _ = kmeans.train(features)
        centroids = kmeans.centroids
        labels = kmeans.index.search(features, 1)[1].reshape(-1)
        visit_probs = self.transition_probs(labels)
        
        pca = PCA(n_components=2)
        fs_pca = pca.fit_transform(features)
        cen_pca = pca.transform(centroids)
        
        # PCA sanity check
        pca2 = PCA(n_components=10)
        pca2.fit(features)
        expvars = pca2.explained_variance_ratio_ * 100
        cumsum = [sum(expvars[:i]) for i in range(1, len(expvars))]
        
        plt.figure(figsize=(8, 6))
        plt.plot(cumsum, linewidth=2, color='b')
        plt.grid(alpha=0.4)
        plt.savefig(os.path.join(self.out_dir, 'mdp_viz', f'pca_variance_cumsum.png'))
        
        plt.figure(figsize=(16, 8))
        plt.subplot(121)
        plt.scatter(fs_pca[:, 0], fs_pca[:, 1], c=labels, s=20, alpha=0.4, cmap='Set1')
        plt.scatter(cen_pca[:, 0], cen_pca[:, 1], c=[i for i in range(centroids.shape[0])], s=120, edgecolors='k', cmap='Set1')
        
        for name, val in visit_probs.items():
            s, s_ = [int(j) for j in name.split('_')]
            cen1, cen2 = cen_pca[s], cen_pca[s_]
            plt.plot([cen1[0], cen2[0]], [cen1[1], cen2[1]], color='k', alpha=0.1)
            
        plt.grid(alpha=0.3)
        plt.title('Clustering by distance', fontsize=15)
        
        plt.subplot(122)
        plt.scatter(fs_pca[:, 0], fs_pca[:, 1], c=actions_buffer, s=20, alpha=0.4, cmap='Set1')
        # plt.scatter(cen_pca[:, 0], cen_pca[:, 1], c=[i for i in range(centroids.shape[0])], s=120, edgecolors='k', cmap='Set1')
        
        # for name, val in visit_probs.items():
        #     s, s_ = [int(j) for j in name.split('_')]
        #     cen1, cen2 = cen_pca[s], cen_pca[s_]
        #     plt.plot([cen1[0], cen2[0]], [cen1[1], cen2[1]], color='k', alpha=0.1)
        
        plt.grid(alpha=0.3)
        plt.title('Clustering by action', fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'mdp_viz', f'100episodes.png'))
        plt.close()
        
    @torch.no_grad()
    def play_on_mdp(self):
        os.makedirs(os.path.join(self.out_dir, 'mdp_viz'), exist_ok=True)
        self.agent.trainable(False)
        features_buffer = []
        actions_buffer = []

        for _ in tqdm(range(100)):        
            episode_done = False 
            total_reward = 0
            step = 0
            
            self.env.new_episode()
            while not episode_done:
                obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
                action = self.agent.select_action(obs)
                reward = self.env.set_action(self.actions[action])
                for _ in range(self.args.frame_skip):
                    self.env.advance_action()
                done = int(self.env.is_episode_finished())
                step += 1
                
                state_fs, _ = self.agent.online_q.encoder(self.agent.preprocess_obs(obs))
                state_fs = state_fs.detach().cpu().numpy()
                features_buffer.append(state_fs)
                actions_buffer.append(action)
                
                if not done:
                    next_obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
                else:
                    next_obs = np.zeros_like(obs, dtype=np.uint8)
                    episode_done = True
                    
        features = np.concatenate(features_buffer, 0).astype(np.float32)
        kmeans = faiss.Kmeans(
            features.shape[1], self.n_actions, niter=20, verbose=False, gpu=torch.cuda.is_available()
        )
        _ = kmeans.train(features)
        centroids = kmeans.centroids
        labels = kmeans.index.search(features, 1)[1].reshape(-1)
        
        pca = PCA(n_components=2)
        fs_pca = pca.fit_transform(features)
        cen_pca = pca.transform(centroids)
        
        labels_, obses = [], []
        rewards, q_preds = [], []
        
        for _ in range(5):
            episode_done = False 
            
            self.env.new_episode()
            while not episode_done:
                frames = [self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)]
                obs = self._warp(frames)
                action = self.agent.select_action(obs)
                reward = self.env.make_action(self.actions[action], self.args.frame_skip)
                done = int(self.env.is_episode_finished())
                rewards.append(reward)
                
                state_fs, _ = self.agent.online_q.encoder(self.agent.preprocess_obs(obs))
                state_fs = state_fs.detach().cpu().numpy().astype(np.float32)
                q_pred = self.agent.online_q(self.agent.preprocess_obs(obs))[0].detach().cpu().numpy().reshape(-1,)[action]
                q_preds.append(q_pred)
                label = kmeans.index.search(state_fs, 1)[1].item()
                labels_.append(label)
                obses.append(frames[-1])
                
                if not done:
                    next_obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
                else:
                    next_obs = np.zeros_like(obs, dtype=np.uint8)
                    episode_done = True
                    
        discounted_sum = lambda seq: sum([seq[i] * self.args.gamma ** i for i in range(len(seq))])
        returns = [discounted_sum(rewards[j:]) for j in range(len(rewards))]
        
        for i in range(len(labels_)):
            plt.figure(figsize=(24, 8))
            plt.subplot(131)
            plt.scatter(fs_pca[:, 0], fs_pca[:, 1], c=labels, s=20, alpha=0.4, cmap='Set1')
            plt.scatter(cen_pca[:, 0], cen_pca[:, 1], 
                        c=np.arange(centroids.shape[0]), 
                        s=[500 if j == labels_[i] else 120 for j in range(centroids.shape[0])], 
                        edgecolors='k', 
                        cmap='Set1')
            plt.grid(alpha=0.3)
            
            plt.subplot(132)
            plt.imshow(obses[i], cmap='gray')
            plt.axis('off')
            
            plt.subplot(133)
            plt.plot(returns[:i], color='b', linewidth=2, label='Actual discounted return')
            plt.plot(q_preds[:i], color='r', linewidth=2, label='Predicted return')
            plt.grid(alpha=0.4)
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_dir, 'mdp_viz', f'episode_play_{i}.png'))
            plt.close()       
        
        
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
        
        self.env, self.actions, self.action_names = env.VizdoomEnv(
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
        
        
class AttentionTrainer:
    
    def __init__(self, args, seed=0):
        self.args = args 
        common.print_args(args)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        
        self.env, self.actions, self.action_names = env.VizdoomEnv(
            name=args.expt_name,
            screen_format=args.screen_format,
            screen_res=args.screen_res
        )
        self.n_actions = len(self.actions)
        
        self.agent = dqn.AttentionDQN(
            input_ch=args.frame_stack, 
            n_actions=self.n_actions, 
            input_res=(args.frame_height, args.frame_width),
            enc_hidden_ch=args.enc_hidden_ch,
            enc_fdim=args.enc_fdim,
            q_hidden_dim=args.q_hidden_dim,
            action_fdim=args.action_fdim,
            attn_model_dim=args.attn_model_dim,
            n_attn_heads=args.n_attn_heads,
            n_attn_layers=args.n_attn_layers,
            gamma=args.gamma,
            eps_max=args.eps_max,
            eps_min=args.eps_min,
            eps_decay_steps=args.eps_decay_steps,
            target_update_interval=args.target_update_interval,
            learning_rate=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            encoder_init_type=args.encoder_init_type,
            encoder_init_ckpt_dir=args.encoder_init_ckpt_dir
        )
        self.memory = memory.ReplayMemory(
            mem_size=args.replay_mem_size, 
            obs_shape=(args.frame_height, args.frame_width, args.frame_stack)
        )
        
        if args.load is not None:
            self.agent.load(args.load)
            self.out_dir = args.load
        elif args.resume is not None:
            self.agent.resume(args.resume)
            self.out_dir = args.resume
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
        self.best_loss = float('inf')
        self.n_heads = args.n_attn_heads
        self.n_layers = args.n_attn_layers
        
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
            if self.args.load is not None or self.args.resume is not None:
                action = self.agent.select_agent_action(obs)
            else:
                action = random.choice(np.arange(self.n_actions))
            reward = self.env.make_action(self.actions[action], self.args.frame_skip)
            done = int(self.env.is_episode_finished())
            if not done:
                next_obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
            else:
                next_obs = np.zeros_like(obs, dtype=np.uint8)
                self.env.new_episode()
            
            self.memory.store(obs, action, reward, next_obs, done)
            common.pbar((step+1)/self.args.mem_init_steps, desc='Progress', status='')
        print()

    def train_episode(self, episode):
        self.agent.trainable(True)
        meter = common.AverageMeter()
        episode_done = False
        step = 0
        
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
            
            self.memory.store(obs, action, reward, next_obs, done)
            step += 1
            
            if step % self.args.mem_replay_interval == 0:
                batch = self.memory.get_batch(self.batch_size)
                loss, acc, _ = self.agent.learn_from_memory(batch)
                meter.add({'AL': loss, 'Acc': acc})

        if episode % self.args.log_interval == 0:
            self.logger.record("E: {:7d} |{}".format(episode, meter.msg()), mode='train')
        if self.log_wandb:
            wandb.log({'episode': episode, 'train_loss': meter.get()['AL'], 'train_acc': meter.get()['Acc']})
            
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
                    
                obs = self.agent.preprocess_obs(obs)
                trg_q, _, _ = self.agent.enc_buffer(obs)
                pred_q, _, _, _ = self.agent.online_q(obs)
                loss, acc = self.agent.loss_fn(pred_q, trg_q)
                meter.add({'AL': loss.item(), 'Acc': acc})
                         
        avg_loss, avg_acc = meter.get()['AL'], meter.get()['Acc']     
        self.logger.record("E: {:7d} | AL: {:.4f} | Acc: {:.4f} |".format(episode, avg_loss, avg_acc), mode='val')
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.agent.save(self.out_dir)
        
        if self.log_wandb:
            wandb.log({'episode': episode, 'val_loss': avg_loss, 'val_acc': avg_acc})
    
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
        
    def gather_attn_maps(self, attn_probs, interpolate_size=None):
        n_patches = attn_probs['layer0'][:, :, :, 1:-self.n_actions].size(-1)
        sidelen = math.sqrt(n_patches)
        assert int(sidelen) == sidelen, f'Attn of size {attn_probs["layer0"][:, :, :, 1:-self.n_actions]}'
        
        layer_action_attn = {}
        for lname, attn in attn_probs.items():
            action_attn = attn[:, :, -self.n_actions:, 1:-self.n_actions]
            layer_action_attn[lname] = {}
            for i in range(self.n_actions):
                attn_map = action_attn[:, :, i, :].view(1, self.n_heads, int(sidelen), int(sidelen))
                attn_map = F.interpolate(attn_map, interpolate_size, mode='bilinear', align_corners=False)
                layer_action_attn[lname][i] = attn_map.squeeze(0).detach().cpu().numpy()
        return layer_action_attn
    
    @torch.no_grad()
    def show_attention_on_frames(self, layer=None):
        if layer is not None:
            assert layer in range(self.n_layers), f'Invalid layer index, should be in {[i for i in range(self.n_layers)]}'
        else:
            layer = self.n_layers - 1
        
        self.agent.trainable(False)
        self.env.set_mode(vzd.Mode.ASYNC_PLAYER)
        self.env.set_screen_format(vzd.ScreenFormat.GRAY8)
        self.env.init()
        
        frame_shape = self.env.get_state().screen_buffer.shape
        resolution = (frame_shape[0], frame_shape[1])

        fig, axarr = plt.subplots(self.n_actions, self.n_heads+1, figsize=(30, 25))
        img_list = []
        
        for ep in range(self.args.spectator_episodes):
            self.env.new_episode()
            steps_taken = 0
            while not self.env.is_episode_finished():
                raw_frames = [self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)]
                frames = self._warp(raw_frames)
                action = self.agent.select_agent_action(frames)
                reward = self.env.set_action(self.actions[action])
                for _ in range(self.args.frame_skip):
                    self.env.advance_action()
                steps_taken += 1
                
                obs = self.agent.preprocess_obs(frames)
                pred_q, _, _, attn_probs = self.agent.online_q(obs)
                attn_probs = self.gather_attn_maps(attn_probs, resolution)
                
                temp_list = []
                for action_id in range(self.n_actions):
                    for head_id in range(self.n_heads+1):
                        if head_id == 0:
                            im = axarr[action_id, head_id].imshow(raw_frames[-1], cmap='gray')
                            im = axarr[action_id, head_id].set_ylabel(self.action_names[action_id])
                            axarr[action_id, head_id].axis('off')
                        else:
                            im = axarr[action_id, head_id].imshow(attn_probs[f'layer{layer}'][action_id][head_id-1], cmap='plasma', vmin=0, vmax=1)
                            axarr[action_id, head_id].axis('off')
                        temp_list.append(im)
                img_list.append(temp_list)
            print('Episode {:2d} | Steps: {:3d}'.format(ep, steps_taken))
            
        anim = animation.ArtistAnimation(fig, img_list, interval=10000, blit=True)
        anim.save(os.path.join(self.out_dir, 'videos', 'attn_viz.mp4'), fps=5)


class VectorizedActionTrainer:
    
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
        
        self.agent = dqn.VectorizedActionDQN(
            input_ch=args.frame_stack,
            n_actions=self.action_size,
            double_dqn=args.double_dqn,
            enc_hidden_ch=args.enc_hidden_ch,
            enc_fdim=args.enc_fdim,
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
    def generate_embeddings(self):
        state_embeds = []
        actions = []
        self.env.new_episode()
        for step in range(self.args.mem_init_steps):
            obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
            state = self.agent.online_q.encoder(self.agent.preprocess_obs(obs))[0].detach().cpu().numpy()
            action = self.agent.select_agent_action(obs)
            _ = self.env.make_action(self.actions[action], self.args.frame_skip)
            done = int(self.env.is_episode_finished())
            if not done:
                next_obs = self._warp([self.env.get_state().screen_buffer for _ in range(self.args.frame_stack)])
            else:
                next_obs = np.zeros_like(obs, dtype=np.uint8)
                self.env.new_episode()
            state_embeds.append(state)
            actions.append(action)
            common.pbar((step+1)/self.args.mem_init_steps, desc='Progress', status='')
        print()
        
        action_embeds = self.agent.online_q.action_embeds.weight.data.detach().cpu().numpy()
        state_embeds = np.concatenate(state_embeds, 0)
        actions = np.array(actions).reshape(-1, 1)
        
        # Store as pandas TSV files
        state_df = pd.DataFrame(state_embeds, index=None, columns=None)
        action_df = pd.DataFrame(actions, index=None, columns=None)
        action_emb_df = pd.DataFrame(action_embeds, index=None, columns=None)
        
        state_df.to_csv(os.path.join(self.out_dir, 'state_embeds.tsv'), sep='\t', index=False, header=False)
        action_df.to_csv(os.path.join(self.out_dir, 'actions.tsv'), sep='\t', index=False, header=False)
        action_emb_df.to_csv(os.path.join(self.out_dir, 'action_embeds.tsv'), sep='\t', index=False, header=False)
        
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
    
    trainer = Trainer(args)
    
    if args.task == 'train':
        trainer.run()
    
    elif args.task == 'record':
        assert args.load is not None, 'Model checkpoint required for recording video'
        trainer.create_video()
    
    elif args.task == 'embed':
        assert args.load is not None, 'Model checkpoint required for generating embeddings'
        trainer.generate_embeddings()
        
    elif args.task == 'attn_viz':
        assert args.load is not None, 'Model checkpoint required for generating attention visualization'
        trainer.show_attention_on_frames(args.viz_layer)
        
    elif args.task == 'mdp':
        assert args.load is not None, 'Model checkpoint required for generating MDP'
        trainer.play_on_mdp()
        
    elif args.task == 'automap':
        assert args.load is not None, 'Model checkpoint required for automap viz'
        trainer.automap_viz()