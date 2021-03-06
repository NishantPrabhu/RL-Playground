
import os
import math
import wandb 
import torch
import faiss
import argparse 
import shutil
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
        
        self.env = env.AtariEnv(
            name=args.expt_name, 
            frame_res=(args.frame_height, args.frame_width), 
            frame_skip=args.frame_skip, 
            stack_frames=args.frame_stack, 
            reset_noops=args.reset_noops, 
            episodic_life=args.episodic_life,
            clip_rewards=args.clip_rewards
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
            self.out_dir = os.path.join('out', 'atari', args.expt_name, dt.now().strftime('%d-%m-%Y_%H-%M'))
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
        
    def init_memory(self):
        obs = self.env.reset()
        for step in range(self.args.mem_init_steps):
            action = self.env.action_space.sample()
            next_obs, reward, done, _ = self.env.step(action)
            reward_scaled = np.sign(reward)
            
            if reward > self.best_train:
                self.best_train = reward
            
            self.memory.store(obs, action, reward_scaled, next_obs, done)
            obs = next_obs if not done else self.env.reset()
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
        
    def train_episode(self, episode):
        self.agent.trainable(True)
        meter = common.AverageMeter()
        episode_done = False 
        total_reward = 0
        step = 0
        
        obs = self.env.reset()
        while not episode_done:
            action = self.agent.select_action(obs)
            next_obs, reward, done, info = self.env.step(action)
            reward_scaled = np.sign(reward)
            total_reward += reward
            step += 1
            
            self.memory.store(obs, action, reward_scaled, next_obs, done)
            if done and info['ale.lives'] == 0:
                episode_done = True
            else:
                obs = next_obs
                
            if step % self.args.mem_replay_interval == 0:
                batch = self.memory.get_batch(self.batch_size)
                loss = self.agent.learn_from_memory(batch)
                meter.add({'loss': loss})
                
        if total_reward > self.best_train:
            self.best_train = round(total_reward)
            
        if episode % self.args.log_interval == 0:
            self.logger.record("Episode {:7d} [reward] {:5d} {} [train_best] {:3d} [val_best] {:3d}".format(
                episode, round(total_reward), meter.msg(), round(self.best_train), round(self.best_val)),
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
            
            obs = self.env.reset()
            while not episode_done:
                action = self.agent.select_agent_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                total_reward += reward
                
                if done and info['ale.lives'] == 0:
                    episode_done = True
                else:
                    obs = next_obs
            meter.add({'reward': total_reward})
                
        avg_reward = meter.get()['reward']
        if avg_reward > self.best_val:
            self.best_val = round(avg_reward)
            self.agent.save(self.out_dir)
              
        self.logger.record("Episode {:7d} [reward] {:5d} [train_best] {:3d} [val_best] {:3d}".format(
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
                action = self.agent.select_agent_action(obs)
                next_obs, reward, done, info = self.env.step(action)
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
    def convert_to_mdp(self, tsne_perplexity=30):
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

        for _ in tqdm(range(5)):        
            episode_done = False 
            total_reward = 0
            step = 0
            obs = self.env.reset()
            
            while not episode_done:
                action = self.agent.select_agent_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                total_reward += reward
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
                    
            print("Total reward: {}".format(total_reward))
                    
        features = np.concatenate(features_buffer, 0).astype(np.float32)
        tsne = TSNE(n_components=2, perplexity=tsne_perplexity)
        fs_tsne = tsne.fit_transform(features)
        pca = PCA(n_components=2)
        fs_pca = pca.fit_transform(features)
        
        df = pd.DataFrame({
            **{f'pca{j}': fs_pca[:, j].reshape(-1) for j in range(fs_pca.shape[1])}, 
            **{f'tsne{j}': fs_tsne[:, j].reshape(-1) for j in range(fs_tsne.shape[1])},
            'impath': filenames, 
            'action': actions_buffer
        })
        df.to_csv("atari_viz_data.csv")
        
        kmeans = faiss.Kmeans(
            features.shape[1], self.n_actions, niter=20, verbose=False, gpu=torch.cuda.is_available()
        )
        _ = kmeans.train(features)
        centroids = kmeans.centroids
        labels = kmeans.index.search(features, 1)[1].reshape(-1)
        visit_probs = self.transition_probs(labels)
        
        
        
        # cen_pca = pca.transform(centroids)
        
        # PCA sanity check
        # pca2 = PCA(n_components=100)
        # pca2.fit(features)
        # expvars = pca2.explained_variance_ratio_ * 100
        # cumsum = [sum(expvars[:i]) for i in range(1, len(expvars))]
        
        # plt.figure(figsize=(8, 6))
        # plt.plot(cumsum, linewidth=2, color='b')
        # plt.grid(alpha=0.4)
        # plt.savefig(os.path.join(self.out_dir, 'mdp_viz', f'pca_variance_cumsum.png'))
        
        plt.figure(figsize=(16, 8))
        plt.subplot(121)
        plt.scatter(fs_pca[:, 0], fs_pca[:, 1], c=labels, s=20, alpha=0.1)
        # plt.scatter(cen_pca[:, 0], cen_pca[:, 1], c=[i for i in range(centroids.shape[0])], s=80, edgecolors='k')
        
        # for name, val in visit_probs.items():
        #     s, s_ = [int(j) for j in name.split('_')]
        #     cen1, cen2 = cen_pca[s], cen_pca[s_]
        #     plt.plot([cen1[0], cen2[0]], [cen1[1], cen2[1]], color='k')
            
        plt.grid(alpha=0.3)
        plt.title('Clustering by distance', fontsize=15)
        
        plt.subplot(122)
        plt.scatter(fs_pca[:, 0], fs_pca[:, 1], c=actions_buffer, s=20, alpha=0.1)
        # plt.scatter(cen_pca[:, 0], cen_pca[:, 1], c=[i for i in range(centroids.shape[0])], s=80, edgecolors='k')
        
        # for name, val in visit_probs.items():
        #     s, s_ = [int(j) for j in name.split('_')]
        #     cen1, cen2 = cen_pca[s], cen_pca[s_]
        #     plt.plot([cen1[0], cen2[0]], [cen1[1], cen2[1]], color='k')
        
        plt.grid(alpha=0.3)
        plt.title('Clustering by action', fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'mdp_viz', f'10episodes_perp={tsne_perplexity}.png'))
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
            
            while not episode_done:
                action = self.agent.select_agent_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                
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
            
            while not episode_done:
                action = self.agent.select_agent_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                rewards.append(reward)
                
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
        
        self.env = env.AtariEnv(
            name=args.expt_name, 
            frame_res=(args.frame_height, args.frame_width), 
            frame_skip=args.frame_skip, 
            stack_frames=args.frame_stack, 
            reset_noops=args.reset_noops, 
            episodic_life=args.episodic_life,
            clip_rewards=args.clip_rewards
        )
        self.n_actions = self.env.action_space.n
        
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
        
    def init_memory(self):
        obs = self.env.reset()
        for step in range(self.args.mem_init_steps):
            action = self.agent.select_agent_action(obs)
            next_obs, reward, done, _ = self.env.step(action)
            if reward > self.best_train:
                self.best_train = reward
            
            self.memory.store(obs, action, reward, next_obs, done)
            obs = next_obs if not done else self.env.reset()
            common.pbar((step+1)/self.args.mem_init_steps, desc='Progress', status='')
            
        total_loss = 0
        random_steps = self.args.mem_init_steps // self.batch_size
        for step in range(random_steps):
            batch = self.memory.get_batch(self.batch_size)
            loss, _, _ = self.agent.learn_from_memory(batch)
            total_loss += loss
            common.pbar((step+1)/random_steps, desc="Learning", status='')
        print()
        avg_loss = total_loss / random_steps
        self.logger.record('[loss] {:.4f} [train_best] {}'.format(avg_loss, self.best_train), mode='train')

    def train_episode(self, episode):
        self.agent.trainable(True)
        meter = common.AverageMeter()
        episode_done = False 
        total_reward = 0
        step = 0
        
        obs = self.env.reset()
        while not episode_done:
            action = self.agent.select_agent_action(obs)
            next_obs, reward, done, info = self.env.step(action)
            total_reward += reward
            step += 1
            
            self.memory.store(obs, action, reward, next_obs, done)
            if done:
                episode_done = True
            else:
                obs = next_obs
                
            if step % self.args.mem_replay_interval == 0:
                batch = self.memory.get_batch(self.batch_size)
                loss, acc, _ = self.agent.learn_from_memory(batch)
                meter.add({'loss': loss, 'accuracy': acc})
                
        if total_reward > self.best_train:
            self.best_train = round(total_reward)
            
        if episode % self.args.log_interval == 0:
            self.logger.record("Episode {:7d} [reward] {:5d} {}".format(
                episode, round(total_reward), meter.msg()),
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
            
            obs = self.env.reset()
            while not episode_done:
                action = self.agent.select_agent_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                total_reward += reward
                
                obs = self.agent.preprocess_obs(obs)
                trg_q, _, _ = self.agent.enc_buffer(obs)
                pred_q, _, _, _ = self.agent.online_q(obs)
                loss, acc = self.agent.loss_fn(pred_q, trg_q)
                meter.add({'loss': loss.item(), 'accuracy': acc})
                
                if done:
                    episode_done = True
                else:
                    obs = next_obs
                    
        avg_loss, avg_acc = meter.get()['loss'], meter.get()['accuracy']
        self.logger.record("Episode {:7d} [loss] {:.4f} [accuracy] {:.4f}".format(episode, avg_loss, avg_acc), mode='val')
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
                action = self.agent.select_agent_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                total_reward += reward
                rec.capture_frame()

                if done and info['ale.lives'] == 0:
                    episode_done = True
                else:
                    obs = next_obs
            
            rec.close()
            rec.enabled = False
            self.env.close()
            self.logger.print("Attempt {:2d} [reward] {}".format(i, total_reward), mode='val')
        
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
        
        resolution = (self.args.frame_height, self.args.frame_width)
        fig, axarr = plt.subplots(self.n_actions, self.n_heads+1, figsize=(30, 25))
        img_list = []
        
        for ep in range(self.args.spectator_episodes):
            obs = self.env.reset()
            episode_finished = False
            steps_taken = 0
            
            while not episode_finished:
                action = self.agent.select_agent_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                steps_taken += 1
                
                pred_q, _, _, attn_probs = self.agent.online_q(obs)
                attn_probs = self.gather_attn_maps(attn_probs, resolution)
                
                temp_list = []
                for action_id in range(self.n_actions):
                    for head_id in range(self.n_heads+1):
                        if head_id == 0:
                            im = axarr[action_id, head_id].imshow(np.asarray(obs)[:, :, -1], cmap='gray')
                            im = axarr[action_id, head_id].set_ylabel(self.action_names[action_id])
                            axarr[action_id, head_id].axis('off')
                        else:
                            im = axarr[action_id, head_id].imshow(attn_probs[f'layer{layer}'][action_id][head_id-1], cmap='plasma', vmin=0, vmax=1)
                            axarr[action_id, head_id].axis('off')
                        temp_list.append(im)
                img_list.append(temp_list)
                
                if done and info['ale.lives'] == 0:
                    episode_finished = True 
                else:
                    obs = next_obs
                    
            print('Episode {:2d} | Steps: {:3d}'.format(ep, steps_taken))
            
        anim = animation.ArtistAnimation(fig, img_list, interval=10000, blit=True)
        anim.save(os.path.join(self.out_dir, 'videos', 'attn_viz.mp4'), fps=5)
        
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

        for _ in tqdm(range(10)):        
            episode_done = False 
            total_reward = 0
            step = 0
            obs = self.env.reset()
            
            while not episode_done:
                action = self.agent.select_agent_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                total_reward += reward
                step += 1
                
                qvals, state_fs, _, _ = self.agent.online_q(self.agent.preprocess_obs(obs))
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
        plt.grid(alpha=0.3)
        plt.title('Clustering by action', fontsize=15)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'mdp_viz', f'100episodes.png'))
        plt.close()
           
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser = cli_args.add_atari_args(parser)
    args = parser.parse_args()
    
    trainer = Trainer(args)
    
    if args.task == 'train':
        trainer.run()
    
    elif args.task == 'record':
        assert args.load is not None, 'Model checkpoint required for recording video'
        trainer.create_video()
        
    elif args.task == 'mdp':
        assert args.load is not None, 'Model checkpoint required for MDP creation'
        trainer.convert_to_mdp(20)