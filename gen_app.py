
import os
import cv2
import time
import math
import wandb 
import torch
import random
import agents
import numpy as np
import pandas as pd
import vizdoom as vzd
import streamlit as st
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from envs import env
from agents import dqn 
from datetime import datetime as dt
from utils import common, memory, cli_args
from gym.wrappers.monitoring.video_recorder import VideoRecorder


ENC_FDIM = 1024
ACTION_SIZE = 8

def initialize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    game, actions, action_names = env.VizdoomEnv(
        name="basic", 
        screen_format="GRAY8",
        screen_res="RES_640X480"
    )
    agent = dqn.DQN(
        input_ch=3,
        n_actions=len(actions),
        double_dqn=True,
        dueling_dqn=True,
        enc_hidden_ch=32,
        enc_fdim=1024,
        q_hidden_dim=512,
        gamma=0.99,
        eps_max=1.0,
        eps_min=0.1,
        eps_decay_steps=500000,
        target_update_interval=10000,
        learning_rate=0.0001,
        max_grad_norm=1.0
    )
    generator = agents.networks.Generator(input_dim=(ENC_FDIM + ACTION_SIZE)).to(device)
    return game, agent, generator

def _warp(frames):
    if frames[0].ndim == 2:
        frames = [np.expand_dims(f, -1) for f in frames]
    elif frames[0].ndim == 3 and frames[0].shape[-1] > 1:
        frames = [np.expand_dims(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), -1) for f in frames]
        
    obs = np.concatenate(frames, -1)
    obs = cv2.resize(obs, (84, 84 ), interpolation=cv2.INTER_AREA)
    return obs


if __name__ == '__main__':
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    game, agent, generator = initialize()
    agent.load('out/vizdoom/basic/27-02-2022_17-06')
    state = torch.load('out/generative/vizdoom/basic/27-02-2022_21-24/gen_state.pth.tar', map_location=device)
    generator.load_state_dict(state['gen'])

    st.title("GAN visualization")
    game.new_episode()
    obs = _warp([game.get_state().screen_buffer for _ in range(3)])    
    noise = agent.online_q.encoder(agent.preprocess_obs(obs))[0].detach().cpu().numpy()
    
    fidx = st.selectbox("Feature index", [i for i in range(ENC_FDIM)])
    fval = st.slider(f"Select value for feature {fidx}", -3.0, 3.0, value=noise[:, fidx].item())
    
    noise[:, fidx] = fval
    action = np.zeros((1, ACTION_SIZE))
    action[:, random.randint(0, ACTION_SIZE-1)] = 1
    inp = np.concatenate([noise, action], -1)
    inp = torch.from_numpy(inp).float().to(device)
    
    with torch.no_grad():
        out = generator(inp)
        out = out.detach().cpu().numpy()
    
    st.header("Result")
    fig = plt.figure(figsize=(6, 6))
    plt.imshow(out[0][0], cmap='gray')
    plt.axis('off')
    st.pyplot(fig)