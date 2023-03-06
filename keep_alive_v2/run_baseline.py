import multiprocessing as mp
import os
import pickle
from Scheduler import Scheduler
from LambdaData import LambdaData
import pandas as pd


def run_multiple_expts():
    record_pth = r"F:\Learning\ColdStart\keep_alive_v2\Record\baseline.csv"
    policies = ["TTL"]
    memstep = 10000
    mems = [i for i in range(80000, 90000, memstep)]
    mems = set(mems)
    numfuncs = [i for i in range(200,500,100)]
    tracelen = 3000

    results = []
    df = pd.DataFrame(columns=('policy', 'mem_capacity', 'num_funcs', 'cold_start_num',
                               'cold_cost', 'mem_cost', 'trace_ptr', 'trace_len'))

    for policy in policies:
        for mem in mems:
            for num_func in numfuncs:
                for char in ["a", "b"]:
                    print("=====Start num_func: {}, mem_capacity: {}, policy: {} =====".format(num_func, mem, policy))
                    df = compare_pols(policy, num_func, char, mem, tracelen, df)

    df.to_csv(record_pth)

def compare_pols(policy, num_func, char, mem, tracelen, df):
    trace_pth = r"F:\Learning\ColdStart\keep_alive_v2\GenTrace\random"
    # print("=====Start num_func: {}, mem_capacity: {}, policy: {} =====".format(num_func, mem, policy))

    L = Scheduler(policy, mem, num_func)
    lambdas, trace = load_trace(num_func, char, trace_pth)
    trace_ptr = 0
    for d, t in trace:
        trace_ptr += 1
        is_sufficient, is_hit, increase_time = L.runActivation(d, t)
        if not is_sufficient:
            #print("Memory is not sufficient")
            break

    L.clear_pool()
    #print("=====End now ptr: {}, cold start num: {}, cold cost: {}, memory cost: {} =====".format(trace_ptr, L.cold_start_num, L.cold_cost, L.mem_cost))
    s = pd.Series({'policy': policy, 'mem_capacity': mem, 'num_funcs': num_func,
                   'cold_start_num': L.cold_start_num, 'cold_cost': L.cold_cost, 'mem_cost': L.mem_cost,
                   'trace_ptr': trace_ptr, 'trace_len': len(trace)})
    df = df.append(s, ignore_index=True)
    print("Finish")
    return df


def load_trace(num_functions, char, trace_path):
    fname = "{}-{}.pckl".format(num_functions, char)
    with open(os.path.join(trace_path, fname), "r+b") as f:
        return pickle.load(f)


if __name__ == '__main__':
    run_multiple_expts()

