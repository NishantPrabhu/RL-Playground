
import os  
import wandb 
import torch
import argparse 
import numpy as np

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
            if reward > self.best_train:
                self.best_train = reward
            
            self.memory.store(obs, action, reward, next_obs, done)
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
        self.logger.record('QL: {:.4f} | BEST_T: {}'.format(avg_loss, self.best_train), mode='train')
        
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
            total_reward += reward
            step += 1
            
            self.memory.store(obs, action, reward, next_obs, done)
            if info['ale.lives'] == 0:
                episode_done = True
            else:
                obs = next_obs if not done else self.env.reset()
                
            if step % self.args.mem_replay_interval == 0:
                batch = self.memory.get_batch(self.batch_size)
                loss = self.agent.learn_from_memory(batch)
                meter.add({'loss': loss})
                
        if total_reward > self.best_train:
            self.best_train = round(total_reward)
            
        if episode % self.args.log_interval == 0:
            self.logger.record("E: {:7d} | R: {:3d} |{} BEST_T: {:3d} | BEST_V: {:3d}".format(
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
                
                if info['ale.lives'] == 0:
                    episode_done = True
                else:
                    obs = next_obs if not done else self.env.reset()
            meter.add({'R': total_reward})
                
        avg_reward = meter.get()['R']
        if avg_reward > self.best_val:
            self.best_val = round(avg_reward)
            self.agent.save(self.out_dir)
              
        self.logger.record("E: {:7d} | R: {:3d} | BEST_T: {:3d} | BEST_V: {:3d}".format(
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

                if info['ale.lives'] == 0:
                    episode_done = True
                else:
                    obs = next_obs if not done else self.env.reset()
            
            rec.close()
            rec.enabled = False
            self.env.close()
            self.logger.print("Attempt {:2d} | R: {}".format(i, total_reward), mode='val')
            
            
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