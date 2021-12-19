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
        msg = "".join([" {}: {:.4f} |".format(key, value) for key, value in metrics.items()])
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
            print(f"| {COLORS['yellow']}TRAIN{COLORS['end']} | {msg}")
        elif mode == 'val':
            print(f"|  {COLORS['blue']}VAL{COLORS['end']}  | {msg}")
        else:
            print(f"{msg}")

    def write(self, msg, mode):
        if mode == "info":
            msg = f"[INFO] {msg}"
        elif mode == "train":
            msg = f"| TRAIN | {msg}"
        elif mode == "val":
            msg = f"|  VAL  | {msg}"
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
    print(text, end="")
    
def print_args(args):
    print("\n---- experiment configuration ----")
    args_ = vars(args)
    for arg, value in args_.items():
        print(f" * {arg} => {value}")
    print("----------------------------------")