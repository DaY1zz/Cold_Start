import argparse
import torch
from Env import Env
from agent.PGAgent import Agent
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=True, type=bool, help="Train mode or not")
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--save_dir', default='results/soccer', type=str)
    parser.add_argument('--train_eps', default=5000, type=int)
    parser.add_argument('--eval_eps', default=1000, type=int)
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--gamma', default=0.90, type=float)
    parser.add_argument('--hidden_dim', default=128, type=int)
    parser.add_argument('--e_greedy', default=0.9, type=float)
    parser.add_argument('--replace_target_iter', default=200, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_funcs', default=20, type=int)

    config = parser.parse_args()
    return config

def train(cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    env = Env(cfg.num_funcs)

    agent = Agent(env, hidden_dim=cfg.hidden_dim, learning_rate=cfg.lr, gamma=cfg.gamma, device=cfg.device)
    agent.train(cfg.train_eps)


if __name__ == '__main__':
    cfg = get_args()
    if cfg.train:
        train(cfg)
    else:
        pass