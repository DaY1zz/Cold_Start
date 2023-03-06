import pickle
import os

import numpy as np

from FaasCache.Container import Container
from FaasCache.LambdaScheduler import LambdaScheduler

TRACE_PATH = ""
NUM_FUNCTIONS = 100
CHAR = "a"
LOG_DIR = ""

def load_trace(num_functions, char, trace_path):
    fname = "{}-{}.pckl".format(num_functions, char)
    with open(os.path.join(trace_path, fname), "r+b") as f:
        return pickle.load(f)

class Env:
    def __init__(self):
        # System metadata
        self.mem_capacity = 40000
        self.policy = "LONG-TTL"
        self.lambdaScheduler = LambdaScheduler(self.policy, self.mem_capacity, NUM_FUNCTIONS, CHAR, LOG_DIR)

        # trace
        self.lambdas, self.trace = load_trace(NUM_FUNCTIONS, CHAR, TRACE_PATH)

        # trace计数器，表示正在处理第 trace_ptr 个trace
        self.trace_ptr = 0

    def reset(self, char):
        self.lambdaScheduler = LambdaScheduler(self.policy, self.mem_capacity, NUM_FUNCTIONS, char, LOG_DIR)

        self.lambdas, self.trace = load_trace(NUM_FUNCTIONS, char, TRACE_PATH)
        self.trace_ptr = 0

    def step(self, action):
        # input: action
        # output: next state, reward, all_done
        cur_trace = self.trace[self.trace_ptr]
        all_done = False
        done = False
        is_hit = False
        reward = 0

        if action == 0:
            # TODO action为0即不分配容器，请求等待一段时间(固定时长？与请求的冷启动时间有关？)
            self.trace[self.trace_ptr][1] += 500

        # 1. execute action and run activation
        done, is_hit, increase_time = self.lambdaScheduler.runActivation(cur_trace[0], cur_trace[1], action)
        if not done:
            # TODO 即使驱逐也内存不足以分配容器，请求需等待一段时间，如果等待如何考虑奖励？
            self.trace[self.trace_ptr][1] += 500

        # 2. get reward
        if done and is_hit:
            reward = 2
        elif done and not is_hit:
            reward = -1

        # 3. get next state
        next_trace = None
        if done:
            next_trace, all_done = self.get_next_trace()
        next_state = self.get_state(next_trace if done else cur_trace)

        return next_state, reward, all_done

    def get_state(self, trace):
        # state:[cold_rate, available_mem, mem_cost, func_id, mem_size, cold_time, warm_time]
        cold_rate  = self.lambdaScheduler.cold_start / self.trace_ptr
        available_mem = self.mem_capacity - self.lambdaScheduler.mem_used
        mem_cost = self.lambdaScheduler.mem_cost

        func_id = trace[0].kind #TODO 转索引
        mem_size = trace[0].mem_size
        cold_time = trace[0].run_time
        warm_time = trace[0].warm_time

        state = np.array(cold_rate, available_mem, mem_cost, func_id, mem_size, cold_time, warm_time)
        return state

    def get_next_trace(self):
        self.trace_ptr += 1
        next_trace = self.trace[self.trace_ptr] if self.trace_ptr < len(self.trace) else None
        all_done = self.trace_ptr == len(self.trace)
        return next_trace, all_done
