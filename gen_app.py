
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
# import vizdoom as vzd
import streamlit as st
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# from envs import env
from agents import dqn 
from datetime import datetime as dt
from utils import common, memory, cli_args
from gym.wrappers.monitoring.video_recorder import VideoRecorder


ENC_FDIM = 1024
ACTION_SIZE = 8

@st.cache
def initialize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # env, actions, action_names = env.VizdoomEnv(
    #     name=args.expt_name, 
    #     screen_format=args.screen_format,
    #     screen_res=args.screen_res
    # )
    generator = agents.networks.Generator(input_dim=(ENC_FDIM + ACTION_SIZE)).to(device)
    state = torch.load('out/generative/27-02-2022_21-24/gen_state.pth.tar', map_location=device)
    generator.load_state_dict(state['gen'])
    return generator


if __name__ == '__main__':
    
    generator = initialize()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    st.title("GAN visualization")
    
    np.random.seed(0)
    noise = np.random.normal(0, 1, size=(1, ENC_FDIM))
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