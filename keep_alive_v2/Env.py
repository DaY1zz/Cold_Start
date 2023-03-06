import pickle
import os
from copy import deepcopy
import numpy as np
from Scheduler import Scheduler
from LambdaData import LambdaData
from Container import Container

TRACE_PATH = ""
CHAR = ""


def load_trace(num_functions, char, trace_path):
    fname = "{}-{}.pckl".format(num_functions, char)
    with open(os.path.join(trace_path, fname), "r+b") as f:
        return pickle.load(f)


class Env:
    def __init__(self, num_funcs):
        # System metadata
        self.mem_capacity = 40000
        self.policy = "Dynamic-TTL"
        self.num_funcs = num_funcs
        self.scheduler_dump = Scheduler(self.policy, self.mem_capacity, self.num_funcs)
        self.scheduler = deepcopy(self.scheduler_dump)

        # trace
        self.lambdas, self.trace = load_trace(self.num_funcs, CHAR, TRACE_PATH)
        self.func_ids = []
        self.func_cold_costs = []
        self.func_mem_sizes = []

        for f in self.lambdas:
            self.func_ids.append(f.kind)
            self.func_cold_costs.append(f.run_time - f.warm_time)
            self.func_mem_sizes.append(f.mem_size)

        # state
        col1 = np.array(self.func_cold_costs).reshape(-1, 1)
        col2 = np.array(self.func_mem_sizes).reshape(-1, 1)
        self.func_attr = np.concatenate((col1, col2), axis=1)  # func_attr: cold_cost, mem_size
        self.func_hit_num = np.zeros((self.num_funcs, 1))
        self.system_attr = np.zeros((2, 1))

        # trace计数器，表示正在处理第 trace_ptr 个trace
        self.trace_ptr = 0

    def reset(self):
        self.scheduler = deepcopy(self.scheduler_dump)
        self.trace_ptr = 0

        self.func_hit_num = np.zeros((self.num_funcs, 1))
        col3 = self.func_hit_num
        col4 = np.zeros((self.num_funcs, 1))
        col5 = np.zeros((self.num_funcs, 1))
        state = np.concatenate((self.func_attr, col3, col4, col5), axis=1)

        self.system_attr[0] = self.mem_capacity
        self.system_attr[1] = 0
        state = np.concatenate((state.reshape(1, -1), self.system_attr.reshape(1, -1)), axis=1)

        return state

    def get_state(self):
        avg_IAT = []
        func_ttl_left = []

        for f in self.lambdas:
            avg_IAT.append(self.scheduler.func_avg_IAT.get(f.kind, 0))
            c: Container = self.scheduler.find_container(f)
            
            if c is None:
                func_ttl_left.append(0)
            else:
                func_ttl_left.append(c.last_access_t + c.keep_alive_TTL - self.scheduler.wall_time)

        # func_attr: cold_cost, mem_size, hit_num, avg_inter_arr_t, ttl_left
        col3 = self.func_hit_num
        col4 = np.array(avg_IAT).reshape(-1, 1)
        col5 = np.array(func_ttl_left).reshape(-1, 1)
        state = np.concatenate((self.func_attr, col3, col4, col5), axis=1)

        # system_attr: mem_left, cold_rate
        self.system_attr[0] = self.mem_capacity - self.scheduler.mem_used
        self.system_attr[1] = self.scheduler.cold_start_num / self.scheduler.trace_ptr

        # shape: 1 * (num_funcs * 5 + 2)
        state = np.concatenate((state.reshape(1, -1), self.system_attr.reshape(1, -1)), axis=1)
        return state

    def step(self, action):
        done = False
        is_hit = True
        reward = 0
        # 1. 执行动作
        self.scheduler.controlContainersTTL(action)

        # 2. 环境变化，获取奖励
        while is_hit:
            if self.trace_ptr == len(self.trace):
                done = True
                cold_rate = self.scheduler.cold_start_num / self.scheduler.trace_ptr
                mem_cost_norm = self.scheduler.mem_cost / (self.mem_capacity * 24 * 60 * 60)
                cold_cost_norm = self.scheduler.cold_cost / self.scheduler.max_cold_cost
                # reward_ep
                reward += -(0.2 * cold_rate + 0.4 * mem_cost_norm + 0.4 * cold_cost_norm) * len(self.trace)
                break

            f, t = self.trace[self.trace_ptr]
            finished, is_hit = self.scheduler.runActivation(f, t)
            if finished:
                done = True
                reward += -len(self.trace)
                break
            if is_hit:
                idx = self.lambdas.index(f)
                self.func_hit_num[idx] += 1
                reward += 1
            self.trace_ptr += 1

        # 3. 冷启动发生，获取新状态
        state_ = self.get_state()
        return state_, reward, done

