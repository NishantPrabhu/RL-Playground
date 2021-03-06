
import os
import gym 
import torch
import dmc2gym
import numpy as np
import vizdoom as vzd
import itertools as it
import gym_minigrid.wrappers as mg_wrappers
from .wrappers import make_env

    
def AtariEnv(name, frame_res, frame_skip, stack_frames, reset_noops, episodic_life, clip_rewards):
    env = gym.make(name)
    env = make_env(env, 'atari', frame_res, frame_skip, stack_frames, reset_noops, episodic_life, clip_rewards)
    return env


def DMControlEnv(name, frame_res, frame_skip, stack_frames, reset_noops, episodic_life, clip_rewards):
    domain_name, task_name = name.split('_')
    env = dmc2gym.make(domain_name, task_name, visualize_reward=False, from_pixels=True, 
                       height=frame_res[0], width=frame_res[1], frame_skip=frame_skip, 
                       channels_first=False)
    env = make_env(env, 'dmc', frame_res, frame_skip, stack_frames, reset_noops, episodic_life, clip_rewards)
    return env


def MiniGridEnv(name, fully_observable=True):
    env = gym.make(name)
    if fully_observable:
        env = mg_wrappers.RGBImgObsWrapper(env)
    else:
        env = mg_wrappers.RGBImgPartialObsWrapper(env)
    env = mg_wrappers.ImgObsWrapper(env)
    env = mg_wrappers.ReseedWrapper(env)
    # env = mg_wrappers.StateBonus(env)
    return env


def VizdoomEnv(name, screen_format='GRAY8', screen_res='RES_640X480'):
    cfg_path = os.path.join('vzd_scenarios', '{}.cfg'.format(name))
    if not os.path.exists(cfg_path):
        raise FileNotFoundError('Could not find vzd_scenarios/{}.cfg'.format(name))
    
    env = vzd.DoomGame()
    env.load_config(cfg_path)
    env.set_window_visible(False)
    env.set_mode(vzd.Mode.PLAYER)
    env.set_screen_format(getattr(vzd.ScreenFormat, screen_format))
    env.set_screen_resolution(getattr(vzd.ScreenResolution, screen_res))
    env.init()
    
    n = env.get_available_buttons_size()
    b_names = {i: str(env.get_available_buttons()[i]).split('.')[1] for i in range(n)}
    actions = [list(a) for a in it.product([0, 1], repeat=n)]
    
    action_names = {}
    for i, a in enumerate(actions):
        a_name = ''
        for j in range(len(a)):
            if a[j] == 1:
                a_name += '{} '.format(b_names[j])
        if len(a_name) == 0:
            a_name = 'NO-OP'
        action_names[i] = a_name

    return env, actions, action_names