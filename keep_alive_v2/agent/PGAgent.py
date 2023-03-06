import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from keep_alive_v2.Env import Env
from keep_alive_v2.LambdaData import LambdaData


# 定义Actor网络，推断下一次发生冷启动是哪个函数，输出概率
class PolicyNet(nn.Module):
    def __init__(self, num_funcs, hidden_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(num_funcs*5+2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_funcs)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        probs = F.softmax(self.fc3(x), dim=1)
        return probs


class Agent:
    def __init__(self, env: Env, hidden_dim, learning_rate, gamma, device):
        self.env = env
        self.gamma = gamma
        self.device = torch.device(device)
        self.policy_net = PolicyNet(env.num_funcs, hidden_dim).to(self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def select_action(self, state):
        state = torch.Tensor(state).to(self.device)
        probs = self.policy_net(state)
        ttls = np.floor(probs.cpu().detach().numpy().ravel()) * 15 * 60 * 1000   # 冷启动概率 * TTL_fixed
        return ttls

    def learn(self, transition_dict):
        reward_list = transition_dict['rewards']
        state_list = transition_dict['states']
        action_list = transition_dict['actions']

        G = 0
        self.optimizer.zero_grad()
        for i in reversed(range(len(reward_list))):  # 从最后一步算起
            reward = reward_list[i]
            state = torch.tensor([state_list[i]],
                                 dtype=torch.float).to(self.device)
            action = torch.tensor([action_list[i]]).view(-1, 1).to(self.device)
            log_prob = torch.log(self.policy_net(state).gather(1, action))  # 这里应该怎么写？？？，没有具体选择的动作
            G = self.gamma * G + reward
            loss = -log_prob * G
            loss.backward()
        self.optimizer.step()

    def store_transition(self, transition_dict, state, action, reward):
        transition_dict['states'].append(state)
        transition_dict['actions'].append(action)
        transition_dict['rewards'].append(reward)

    def train(self, num_episodes):
        for i_episode in range(num_episodes):
            state = self.env.reset()
            ep_reward = 0
            transition_dict = {'states': [], 'actions': [], 'rewards': []}

            while True:
                action = self.select_action(state)
                state_, reward, done = self.env.step(action)
                self.store_transition(transition_dict, state, action, reward)     # TODO 这里的action存什么呢？冷启动的函数索引？
                ep_reward += reward

                if done:
                    self.learn(transition_dict)
                    break

                state = state_















