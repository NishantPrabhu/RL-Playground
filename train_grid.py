
import os
import cv2
import math
import wandb 
import torch
import faiss
import shutil
import argparse 
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from envs import env
from tqdm import tqdm
from agents import dqn 
from faiss import Kmeans
from PIL import Image
from datetime import datetime as dt 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
        
        self.env = env.MiniGridEnv(
            name=args.expt_name,
            fully_observable=args.fully_observable
        )
        self.agent = dqn.DQN(
            input_ch=args.frame_stack,
            n_actions=self.env.action_space.n,
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
        self.n_actions = self.env.action_space.n
        
        if args.load is not None:
            self.agent.load(args.load)
            self.out_dir = args.load
        else:        
            self.out_dir = os.path.join('out', 'minigrid', args.expt_name, dt.now().strftime('%d-%m-%Y_%H-%M'))
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
        self.best_train, self.best_val = -100, -100
        
    def init_memory(self):
        obs = self.env.reset()
        obs = self._warp(obs)
        for step in range(self.args.mem_init_steps):
            action = self.env.action_space.sample()
            next_obs, reward, done, _ = self.env.step(action)
            next_obs = self._warp(next_obs)
            
            if reward > self.best_train:
                self.best_train = reward
            
            self.memory.store(obs, action, reward, next_obs, done)
            obs = next_obs if not done else self._warp(self.env.reset())
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
        self.logger.record('[loss] {:.4f} [train_best] {}'.format(avg_loss, self.best_train), mode='train')
        
    def _warp(self, frame):
        return cv2.resize(frame, (args.frame_width, args.frame_height), cv2.INTER_AREA)
        
    def train_episode(self, episode):
        self.agent.trainable(True)
        meter = common.AverageMeter()
        episode_done = False 
        total_reward = 0
        step = 0
        
        obs = self.env.reset()
        obs = self._warp(obs)
        while not episode_done:
            action = self.agent.select_action(obs)
            next_obs, reward, done, info = self.env.step(action)
            # if "Lava" not in self.args.expt_name and "DistShift" not in self.args.expt_name:
            reward = reward * 10 - 0.001 * (1 - int(done))
            next_obs = self._warp(next_obs)
            total_reward += reward
            step += 1
            
            self.memory.store(obs, action, reward, next_obs, done)
            if done:
                episode_done = True
            else:
                obs = next_obs
                
            if step % self.args.mem_replay_interval == 0:
                batch = self.memory.get_batch(self.batch_size)
                loss = self.agent.learn_from_memory(batch)
                meter.add({'loss': loss})
                
        if total_reward > self.best_train:
            self.best_train = round(total_reward)
            
        if episode % 50 == 0:
            self.agent.save(self.out_dir)
            
        if episode % self.args.log_interval == 0:
            self.logger.record("Episode {:7d} [reward] {:.4f} [steps] {} {} [train_best] {:.4f} [val_best] {:.4f}".format(
                episode, total_reward, step, meter.msg(), self.best_train, self.best_val),
                mode='train'
            )
        if self.log_wandb:
            wandb.log({'episode': episode, 'reward': total_reward, **meter.get()})

    @torch.no_grad()
    def evaluate(self, episode):
        self.agent.trainable(False)
        meter = common.AverageMeter()
        step = 0
            
        for _ in range(self.args.eval_episodes):
            episode_done = False 
            total_reward = 0
            
            obs = self.env.reset()
            obs = self._warp(obs)
            while not episode_done:
                action = self.agent.select_agent_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                # if "Lava" not in self.args.expt_name and "DistShift" not in self.args.expt_name:
                reward = reward * 10 - 0.001 * (1-int(done))
                next_obs = self._warp(next_obs)
                total_reward += reward
                step += 1
                
                if done:
                    episode_done = True
                else:
                    obs = next_obs
            meter.add({'reward': total_reward})
                
        avg_reward = meter.get()['reward']
        if avg_reward > self.best_val:
            self.best_val = round(avg_reward)
            # self.agent.save(self.out_dir)
              
        self.logger.record("Episode {:7d} [reward] {:.4f} [steps] {} [train_best] {:.4f} [val_best] {:.4f}".format(
            episode, avg_reward, step, self.best_train, self.best_val),
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
            obs = self._warp(obs)
            rec.capture_frame()
            while not episode_done:
                action = self.agent.select_agent_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                # if "Lava" not in self.args.expt_name and "DistShift" not in self.args.expt_name:
                reward = reward * 10 - 0.001 * (1-int(done))
                next_obs = self._warp(next_obs)
                total_reward += reward
                rec.capture_frame()

                if done:
                    episode_done = True
                else:
                    obs = next_obs
            
            rec.close()
            rec.enabled = False
            self.env.close()
            self.logger.print("Attempt {:2d} | R: {}".format(i, total_reward), mode='val')
            
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
        if not os.path.exists('assets'):
            os.makedirs('assets')
        
        if len(os.listdir('assets')) > 0:
            shutil.rmtree('assets')
            os.makedirs('assets')
            
        os.makedirs(os.path.join(self.out_dir, 'mdp_viz'), exist_ok=True)
        self.agent.trainable(False)
        features_buffer = []
        actions_buffer = []
        filenames = []
        count = 0

        for _ in tqdm(range(100)):        
            episode_done = False 
            total_reward = 0
            step = 0
            obs = self.env.reset()
            obs = self._warp(obs)
            
            while not episode_done:
                action = self.agent.select_agent_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                reward_scaled = reward - 0.01 * (1-int(done))
                next_obs = self._warp(next_obs)
                total_reward += reward_scaled
                step += 1
                
                state_fs, _ = self.agent.online_q.encoder(self.agent.preprocess_obs(obs))
                state_fs = state_fs.detach().cpu().numpy()
                features_buffer.append(state_fs)
                actions_buffer.append(action)
                
                frame_img = Image.fromarray(np.asarray(obs)[:, :, -1])
                frame_img.save(f'assets/{count}.png', format='PNG')
                filenames.append(f'{count}.png')
                count += 1
                
                if done:
                    episode_done = True
                else:
                    obs = next_obs
                    
        features = np.concatenate(features_buffer, 0).astype(np.float32)
        tsne = TSNE(n_components=2, perplexity=100)
        fs_tsne = tsne.fit_transform(features)
        pca = PCA(n_components=2)
        fs_pca = pca.fit_transform(features)
        
        df = pd.DataFrame({
            **{f'pca{j}': fs_pca[:, j].reshape(-1) for j in range(fs_pca.shape[1])}, 
            **{f'tsne{j}': fs_tsne[:, j].reshape(-1) for j in range(fs_tsne.shape[1])},
            'impath': filenames, 
            'action': actions_buffer
        })
        df.to_csv("minigrid_viz_data.csv")
        
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
        pca2 = PCA(n_components=100)
        pca2.fit(features)
        expvars = pca2.explained_variance_ratio_ * 100
        cumsum = [sum(expvars[:i]) for i in range(1, len(expvars))]
        
        plt.figure(figsize=(8, 6))
        plt.plot(cumsum, linewidth=2, color='b')
        plt.grid(alpha=0.4)
        plt.savefig(os.path.join(self.out_dir, 'mdp_viz', f'pca_variance_cumsum.png'))
        
        plt.figure(figsize=(16, 8))
        plt.subplot(121)
        plt.scatter(fs_pca[:, 0], fs_pca[:, 1], c=labels, s=20, alpha=0.1)
        plt.scatter(cen_pca[:, 0], cen_pca[:, 1], c=[i for i in range(centroids.shape[0])], s=80, edgecolors='k')
        
        for name, val in visit_probs.items():
            s, s_ = [int(j) for j in name.split('_')]
            cen1, cen2 = cen_pca[s], cen_pca[s_]
            plt.plot([cen1[0], cen2[0]], [cen1[1], cen2[1]], color='k')
            
        plt.grid(alpha=0.3)
        plt.title('Clustering by distance', fontsize=15)
        
        plt.subplot(122)
        plt.scatter(fs_pca[:, 0], fs_pca[:, 1], c=actions_buffer, s=20, alpha=0.1)
        plt.scatter(cen_pca[:, 0], cen_pca[:, 1], c=[i for i in range(centroids.shape[0])], s=80, edgecolors='k')
        
        for name, val in visit_probs.items():
            s, s_ = [int(j) for j in name.split('_')]
            cen1, cen2 = cen_pca[s], cen_pca[s_]
            plt.plot([cen1[0], cen2[0]], [cen1[1], cen2[1]], color='k')
        
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

        for _ in tqdm(range(10)):        
            episode_done = False 
            obs = self.env.reset()
            obs = self._warp(obs)
            
            while not episode_done:
                action = self.agent.select_agent_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                next_obs = self._warp(next_obs)

                state_fs, _ = self.agent.online_q.encoder(self.agent.preprocess_obs(obs))
                state_fs = state_fs.detach().cpu().numpy()
                features_buffer.append(state_fs)
                actions_buffer.append(action)
                
                if done:
                    episode_done = True
                else:
                    obs = next_obs
                    
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
        
        for _ in range(1):        
            episode_done = False 
            obs = self.env.reset()
            obs = self._warp(obs)
            
            while not episode_done:
                action = self.agent.select_agent_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                reward_scaled = reward * 100 - 1 * (1 - int(done))
                next_obs = self._warp(next_obs)
                rewards.append(reward_sclaed)
                
                state_fs, _ = self.agent.online_q.encoder(self.agent.preprocess_obs(obs))
                state_fs = state_fs.detach().cpu().numpy()
                q_pred = self.agent.online_q(self.agent.preprocess_obs(obs))[0].detach().cpu().numpy().reshape(-1)[action]
                q_preds.append(q_pred)
                obses.append(np.asarray(obs)[:, :, -1])
                label = kmeans.index.search(state_fs, 1)[1].item()
                labels_.append(label)
                
                if done:
                    episode_done = True
                else:
                    obs = next_obs
                    
        discounted_sum = lambda seq: sum([seq[i] * self.args.gamma ** i for i in range(len(seq))])
        returns = [discounted_sum(rewards[j:]) for j in range(len(rewards))]
        
        for i in range(len(labels_)):
            if i % 5 == 0:
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
                


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser = cli_args.add_minigrid_args(parser)
    args = parser.parse_args()
    
    trainer = Trainer(args)
    
    if args.task == 'train':
        trainer.run()
    
    elif args.task == 'record':
        assert args.load is not None, 'Model checkpoint required for recording video'
        trainer.create_video()
        
    elif args.task == 'mdp':
        assert args.load is not None, 'Model checkpoint required for MDP creation'
        trainer.convert_to_mdp()