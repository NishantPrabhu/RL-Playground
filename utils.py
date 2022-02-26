
import os 
import yaml 
import torch
import random 
import logging 
import argparse
import numpy as np

COLORS = {
    "yellow": "\x1b[33m", 
    "blue": "\u001b[36m", 
    "green": "\x1b[32m",
    "red": "\x1b[33m", 
    "end": "\033[0m"
}


class AverageMeter:

    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}

    def add(self, metrics: dict):
        if len(self.metrics) == 0:
            self.metrics = {key: [value] for key, value in metrics.items()}
        else:
            for key, value in metrics.items():
                if key in self.metrics.keys():
                    self.metrics[key].append(value)
                else:
                    self.metrics[key] = [value]
    
    def get(self):
        return {key: np.mean(value) for key, value in self.metrics.items()}

    def msg(self):
        metrics = self.get()
        msg = " ".join(["[{}] {:.4f}".format(key, value) for key, value in metrics.items()])
        return msg


class Logger:

    def __init__(self, output_dir):
        [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
        logging.basicConfig(
            level = logging.INFO,
            format = "%(message)s",
            handlers = [logging.FileHandler(os.path.join(output_dir, "logs.txt"))])

    def print(self, msg, mode=""):
        if mode == "info":
            print(f"[{COLORS['green']}INFO{COLORS['end']}] {msg}")
        elif mode == 'train':
            print(f"[{COLORS['yellow']}TRAIN{COLORS['end']}] {msg}")
        elif mode == 'val':
            print(f"[ {COLORS['blue']}VAL{COLORS['end']} ] {msg}")
        else:
            print(f"{msg}")

    def write(self, msg, mode):
        if mode == "info":
            msg = f"[INFO] {msg}"
        elif mode == "train":
            msg = f"[TRAIN] {msg}"
        elif mode == "val":
            msg = f"[ VAL ] {msg}"
        logging.info(msg)

    def record(self, msg, mode):
        self.print(msg, mode)
        self.write(msg, mode)


def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def pbar(progress=0, desc="Progress", status="", barlen=20):
    status = status.ljust(30)
    if progress == 1:
        status = "{}".format(status.ljust(30))
    length = int(round(barlen * progress))
    text = "\r{}: [{}] {:.2f}% {}".format(
        desc, COLORS["green"] + "="*(length-1) + ">" + COLORS["end"] + " " * (barlen-length), progress * 100, status  
    ) 
    print(text, end="" if progress < 1 else "\n")
    
def print_args(args):
    print("\n---- experiment configuration ----")
    args_ = vars(args)
    for arg, value in args_.items():
        print(f" * {arg} => {value}")
    print("----------------------------------")
    
def parse_args(parser):
    parser.add_argument('--task', required=True, type=str)
    parser.add_argument('--expt_name', required=True, type=str)
    parser.add_argument('--load', default=None, type=str)
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--encoder_type', default='vit', type=str)
    parser.add_argument('--frame_height', default=84, type=int)
    parser.add_argument('--frame_width', default=84, type=int)
    parser.add_argument('--frame_stack', default=4, type=int)
    parser.add_argument('--frame_skip', default=4, type=int)
    parser.add_argument('--reset_noops', default=30, type=int)
    parser.add_argument('--episodic_life', default=True, action='store_true')
    parser.add_argument('--clip_rewards', default=False, action='store_true')
    parser.add_argument('--double_dqn', default=True, action='store_true')
    parser.add_argument('--duel_dqn', default=True, action='store_true')
    parser.add_argument('--n_layers', default=2, type=int)
    parser.add_argument('--n_heads', default=4, type=int)
    parser.add_argument('--model_dim', default=128, type=int)
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--n_embeds', default=1024, type=int)
    parser.add_argument('--add_action_embeds', default=True, action='store_true')
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--eps_max', default=1.0, type=float)
    parser.add_argument('--eps_min', default=0.01, type=float)
    parser.add_argument('--eps_decay_steps', default=100_000, type=int)
    parser.add_argument('--trg_update_interval', default=10_000, type=int)
    parser.add_argument('--log_interval', default=1, type=int)
    parser.add_argument('--base_lr', default=0.001, type=float)
    parser.add_argument('--min_lr', default=1e-10, type=float)
    parser.add_argument('--lr_decay_steps', default=10_000_000, type=int)
    parser.add_argument('--lr_decay_factor', default=0.5, type=float)
    parser.add_argument('--weight_decay', default=1e-05, type=float)
    parser.add_argument('--max_grad_norm', default=5.0, type=float)
    parser.add_argument('--mem_init_steps', default=5_000, type=int)
    parser.add_argument('--replay_mem_size', default=100_000, type=int)
    parser.add_argument('--replay_batch_size', default=32, type=int)
    parser.add_argument('--exp_replay_interval', default=1, type=int)
    parser.add_argument('--train_episodes', default=100_000, type=int)
    parser.add_argument('--eval_interval', default=1_000, type=int)
    parser.add_argument('--eval_episodes', default=10, type=int)
    return parser