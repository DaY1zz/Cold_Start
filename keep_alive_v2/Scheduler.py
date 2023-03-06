import sys

import numpy as np
from math import floor, isnan
import heapq
from collections import defaultdict
from Function import Function
from Container import Container


class Scheduler:
    hist_num_cols = [i for i in range(4 * 60)]

    def __init__(self, policy: str = "TTL", mem_capacity: int = 32000, num_funcs: int = 10):
        self.mem_capacity = mem_capacity
        self.mem_used = 0
        self.eviction_policy = policy
        self.wall_time = 0
        self.finish_times = []
        self.running_c = dict()
        self.ContainerPool = []

        self.evdict = defaultdict(int)
        self.capacity_misses = defaultdict(int)
        self.TTL = 10 * 60 * 1000  # 10 minutes in ms
        self.Long_TTL = 2 * 60 * 60 * 1000  # 2 hours in ms

        self.IT_histogram = dict()
        self.last_seen = dict()  # func-name : last seen time
        self.wellford = dict()  # func-name : aggregate
        self.histTTL = dict()  # func-name : time to live
        self.histPrewarm = dict()  # func-name : prewarm time
        self.rep = dict()  # func-name : LambdaData; used to prewarm containers
        self.func_avg_IAT = dict()
        self.invocation_num = dict()

        self.cold_start_num = 0  # number of cold starts
        self.mem_cost = 0  # memory cost = mem_size * keep_alive_time
        self.cold_cost = 0  # cold cost = sum of cold_time - warm_time
        self.max_cold_cost = 0

        self.trace_ptr = 0

    ##############################################################
    # 对已有函数容器的TTL进行更新，若没有则创建新容器，如果创建新容器失败则返回False
    def controlContainersTTL(self, funcs, ttls):
        for f, ttl in zip(funcs, ttls):
            c = self.find_container(f)
            if c is not None:
                c.keep_alive_TTL += ttl
            else:
                if ttl != 0:
                    c = Container(f, ttl)
                    added = self.AddToPool(c)
                    if not added:
                        print("Not enough memory to add container")
                    return False
        return True

    ##############################################################
    # 所有请求处理完后，进行内存开销清算
    def clear_pool(self):
        # cleanup_finished
        t = sys.maxsize
        finished = []
        for c in self.running_c:
            (start_t, fin_t) = self.running_c[c]
            if t >= fin_t:
                finished.append(c)

        for c in finished:
            # HIST方法：每个函数执行完后容器移除，prewarm时间后再加载进内存
            if c.metadata.kind in self.histPrewarm and self.histPrewarm[c.metadata.kind] != 0:
                self.ContainerPool.remove(c)
                (start_t, fin_t) = self.running_c[c]
                self.mem_cost += c.metadata.mem_size * (
                            (fin_t - start_t) / 1000)
            del self.running_c[c]

        # 如果是HIST，这一步是不会进行的
        for i in range(len(self.ContainerPool)-1, -1, -1):
            self.RemoveFromPool(self.ContainerPool[i], "Clear")

    ##############################################################
    # 查找容器池中是否有对应的容器，如果有则检查其TTL，未超时则返回该容器，超时则remove返回None，如果没有则返回None
    def find_container(self, f: Function):
        """ search through the containerpool for matching container """
        if len(self.ContainerPool) == 0:
            return None
        containers_for_the_lambda = [x for x in self.ContainerPool if
                                     (x.metadata == f)]  # 容器可以并发处理多个请求，所以这里不需要判断容器是否在运行，Pool中有对应容器即可
        # for const-ttl, filter here, and remove.
        if self.eviction_policy == "TTL" and containers_for_the_lambda != []:
            # All the old containers, get rid of them
            fresh_containers = self.PurgeOldTTL(
                containers_for_the_lambda)  # This also deletes the containers from containerpool
            containers_for_the_lambda = fresh_containers

        if self.eviction_policy == "Dynamic-TTL" and containers_for_the_lambda != []:
            # All the old containers, get rid of them
            fresh_containers = self.PurgeOldDynamicTTL(
                containers_for_the_lambda)  # This also deletes the containers from containerpool
            containers_for_the_lambda = fresh_containers

        if self.eviction_policy == "HIST" and containers_for_the_lambda != []:
            fresh_containers = self.PurgeOldHist(
                containers_for_the_lambda)  # This also deletes the containers from containerpool
            containers_for_the_lambda = fresh_containers

        if containers_for_the_lambda == []:
            return None
        else:
            return containers_for_the_lambda[0]
        # Just return the first element.
        # Later on, maybe sort by something? Priority? TTL ?

    ##############################################################

    def pool_stats(self):
        pool = self.ContainerPool  # Is a heap
        sdict = defaultdict(int)
        for c in pool:
            k = c.metadata.kind
            sdict[k] += 1

        return sdict

    ##############################################################

    def container_clones(self, c):
        return [x for x in self.ContainerPool if x.metadata == c.metadata]

    ##############################################################

    def calc_priority(self, c):
        return c.last_access_t

    ##############################################################

    def checkfree(self, c):
        return c.metadata.mem_size + self.mem_used <= self.mem_capacity

    ##############################################################
    # 新建容器进入内存
    def AddToPool(self, c: Container, prewarm: bool = False, at_time=None):
        if not prewarm and at_time is not None:
            raise Exception("Can only add container at self.wall_time when not prewarming")

        mem_size = c.metadata.mem_size
        if mem_size + self.mem_used <= self.mem_capacity:
            # Have free space
            self.mem_used = self.mem_used + mem_size

            if prewarm and at_time is not None:
                c.last_access_t = at_time
                c.keep_alive_start_t = at_time
            else:
                c.last_access_t = self.wall_time
                c.keep_alive_start_t = self.wall_time
            c.Priority = self.calc_priority(c)
            self.ContainerPool.append(c)  # 去除堆排序
            return True
        else:
            print("Not enough space for memsize, used, capacity.", mem_size, self.mem_used, self.mem_capacity)
            return False

    ##############################################################

    def RemoveFromPool(self, c: Container, reason: str):
        if c in self.running_c:
            # raise Exception("Cannot remove a running container")
            #print("Cannot remove a running container, reason: ", reason)  # 这里无所谓，后续运行完后容器还是会被移除，运行时间的数量级不会影响内存开销
            return
        self.ContainerPool.remove(c)
        self.mem_used -= c.metadata.mem_size

        print("Remove container from pool, reason: ", reason, "wall time:", self.wall_time/(1000*60), "pool size:", len(self.ContainerPool))

        # 容器被移除时，计算其内存开销
        if self.eviction_policy == "TTL":
            self.mem_cost += c.metadata.mem_size * ((c.last_access_t + self.TTL - c.keep_alive_start_t) / 1000)
        elif self.eviction_policy == "Dynamic-TTL":
            self.mem_cost += c.metadata.mem_size * ((c.last_access_t + c.keep_alive_TTL - c.keep_alive_start_t) / 1000)
        elif self.eviction_policy == "HIST":
            self.mem_cost += c.metadata.mem_size * ((
                        c.last_access_t + self.histTTL[c.metadata.kind] - c.keep_alive_start_t) / 1000)

    ##############################################################

    def PurgeOldDynamicTTL(self, container_list):
        """ Return list of still usable containers after purging those older than TTL """

        kill_list = [c for c in container_list
                     if c.last_access_t + c.keep_alive_TTL < self.wall_time]

        # Aargh this is duplicated from Eviction. Hard to merge though.
        for k in kill_list:
            self.RemoveFromPool(k, "TTL-purge")
            kind = k.metadata.kind
            self.evdict[kind] += 1

        # This is just the inverse of kill_list. Crummy way to do this, but YOLO
        valid_containers = [c for c in container_list
                            if c.last_access_t + c.keep_alive_TTL >= self.wall_time]

        return valid_containers

    ##############################################################

    def PurgeOldTTL(self, container_list):
        """ Return list of still usable containers after purging those older than TTL """

        kill_list = [c for c in container_list
                     if c.last_access_t + self.TTL < self.wall_time]

        # Aargh this is duplicated from Eviction. Hard to merge though.
        for k in kill_list:
            self.RemoveFromPool(k, "TTL-purge")
            kind = k.metadata.kind
            self.evdict[kind] += 1

        # This is just the inverse of kill_list. Crummy way to do this, but YOLO
        valid_containers = [c for c in container_list
                            if c.last_access_t + self.TTL >= self.wall_time]

        return valid_containers

    ##############################################################

    def cache_miss(self, f: Function):

        c = Container(f)
        added = self.AddToPool(c)

        if not added:
            print("actual in-use memory", sum([k.metadata.mem_size for k in self.ContainerPool]))
            print("pool size", len(self.ContainerPool))
            return None

        return c

    ##############################################################

    def cleanup_finished(self):
        """ Go through running containers, remove those that have finished """
        t = self.wall_time
        finished = []
        for c in self.running_c:
            (start_t, fin_t) = self.running_c[c]
            if t >= fin_t:
                finished.append(c)

        for c in finished:
            del self.running_c[c]
            if c.metadata.kind in self.histPrewarm and self.histPrewarm[c.metadata.kind] != 0:
                self.RemoveFromPool(c, "HIST-prewarm")

        # We'd also like to set the container state to WARM (or atleast Not running.)
        # But hard to keep track of the container object references?
        return len(finished)

    ##############################################################

    def runActivation(self, f: Function, t=0):

        # First thing we want to do is queuing delays?
        # Also some notion of concurrency level. No reason that more cannot be launched with some runtime penalty...
        # Let's assume infinite CPUs and so we ALWAYS run at time t
        is_hit = False
        increase_time = 0

        self.wall_time = t
        self.cleanup_finished()

        self.PreWarmContainers()
        self.track_activation(f)

        # Concurrency check can happen here. If len(running_c) > CPUs, put in the queue.
        # Could add fake 'check' entries corresponding to finishing times to check and drain the queue...

        c = self.find_container(f)
        if c is None:
            # Launch a new container since we didnt find one for the metadata ...
            c = self.cache_miss(f)
            if c is None:
                # insufficient memory
                self.capacity_misses[f.kind] += 1
                self.trace_ptr += 1
                return False, is_hit
            c.run()
            # Need to update priority here?
            processing_time = f.run_time
            increase_time = (f.run_time - f.warm_time) / 1000  # 冷启动导致的开销，否则为0
            self.running_c[c] = (t, t + processing_time)
            self.cold_start_num += 1

        else:
            # Strong assumption. If we can find the container, it is warm.
            c.run()
            processing_time = f.warm_time  # d.run_time - d.warm_time
            self.running_c[c] = (t, t + processing_time)
            is_hit = True

        # update the priority here!!
        c.last_access_t = self.wall_time
        new_prio = self.calc_priority(c)  # , update=True)
        c.Priority = new_prio
        # Since frequency is cumulative, this will bump up priority of this specific container
        # rest of its clones will be still low prio. We should recompute all clones priority

        self.cold_cost += increase_time
        self.max_cold_cost += f.run_time - f.warm_time
        self.trace_ptr += 1

        # Now rebalance the heap and update container access time
        return True, is_hit

    ### HIST方法
    ##############################################################

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    # For a new value newValue, compute the new count, new mean, the new M2.
    # mean accumulates the mean of the entire dataset
    # M2 aggregates the squared distance from the mean
    # count aggregates the number of samples seen so far
    def well_update(self, existingAggregate, newValue):
        (count, mean, M2) = existingAggregate
        count += 1
        delta = newValue - mean
        mean += delta / count
        delta2 = newValue - mean
        M2 += delta * delta2

        return (count, mean, M2)

    # Retrieve the mean, variance and sample variance from an aggregate
    def well_finalize(self, existingAggregate):
        (count, mean, M2) = existingAggregate
        if count < 2:
            return float('nan'), float('nan'), float('nan')
        else:
            (mean, variance, sampleVariance) = (mean, M2 / count, M2 / (count - 1))
        return (mean, variance, sampleVariance)

    ##############################################################

    def _find_precentile(self, cdf, percent, head=False):
        """ Returns the last whole bucket (minute) before the percentile """
        for i, value in enumerate(cdf):
            if percent < value:
                if head:
                    return max(0, i - 1)
                else:
                    return min(i + 1, len(cdf))
        return len(cdf)

    def track_activation(self, f: Function):
        if self.eviction_policy != "HIST" or self.eviction_policy != "Dynamic-TTL":
            return

        IAT = 0
        four_hours_in_mins = 4 * 60
        t = self.wall_time
        if f.kind in self.last_seen:
            IAT = self.wall_time - self.last_seen[f.kind]
            self.invocation_num[f.kind] += 1
            self.func_avg_IAT[f.kind] += (IAT - self.func_avg_IAT[f.kind]) / self.invocation_num[f.kind]  # 计算对应函数的平均IAT
        else:
            self.func_avg_IAT[f.kind] = 0
            self.invocation_num[f.kind] = 0
        self.last_seen[f.kind] = t

        if not f.kind in self.IT_histogram:
            # never lambda seen before
            self.rep[f.kind] = f

            self.IT_histogram[f.kind] = np.zeros(four_hours_in_mins)
            self.wellford[f.kind] = (0, 0, 0)
            # default TTL is 2 hours in miliseconds
            self.histTTL[f.kind] = 2 * 60 * 60 * 1000
            self.histPrewarm[f.kind] = 0
            return

        minute = floor(IAT / (60 * 1000))
        if minute >= four_hours_in_mins:
            # don't track IAT over 4 hours
            return

        self.wellford[f.kind] = self.well_update(self.wellford[f.kind], minute)
        self.IT_histogram[f.kind][minute] += 1
        mean, variance, sampleVariance = self.well_finalize(self.wellford[f.kind])
        if not isnan(mean):
            data = self.IT_histogram[f.kind]
            cdf = np.cumsum(data / data.sum())
            head = self._find_precentile(cdf, 0.05, head=True)
            tail = self._find_precentile(cdf, 0.99)

            if mean == 0 or variance / mean <= 2:
                self.histTTL[f.kind] = tail * 60 * 1000 * 1.1  # 10% increase margin
                self.histPrewarm[f.kind] = head * 60 * 1000 * 0.9  # 10% decrease margin
            else:
                # default TTL is 2 hours in miliseconds
                self.histTTL[f.kind] = 2 * 60 * 60 * 1000
                self.histPrewarm[f.kind] = 0  # default to not unload
        else:
            # default TTL is 2 hours in miliseconds
            self.histTTL[f.kind] = 2 * 60 * 60 * 1000
            self.histPrewarm[f.kind] = 0  # default to not unload

    ##############################################################

    def PurgeOldHist(self, container_list):
        """ Return list of still usable containers after purging those older than TTL """
        kill_list = [c for c in container_list
                     if c.last_access_t + self.histTTL[c.metadata.kind] < self.wall_time]

        for k in kill_list:
            self.RemoveFromPool(k, "HIST-TTL-purge")
            kind = k.metadata.kind
            self.evdict[kind] += 1

        # This is just the inverse of kill_list. Crummy way to do this, but YOLO
        valid_containers = [c for c in container_list
                            if c.last_access_t + self.histTTL[c.metadata.kind] >= self.wall_time]

        return valid_containers

    ##############################################################

    def PreWarmContainers(self):
        """Warm containers before incoming activation to mimic it happening at the actual time """
        if self.eviction_policy != "HIST":
            return
        to_warm = [kind for kind, prewarm in self.histPrewarm.items() if prewarm != 0 and prewarm + self.last_seen[
            kind] >= self.wall_time]  # 这里是不是写错了，应该是 prewarm + self.last_seen[kind] < self.wall_time
        for kind in to_warm:
            c = Container(self.rep[kind])
            at = self.histPrewarm[kind] + self.last_seen[kind]
            at = min(at, self.wall_time)
            self.AddToPool(c=c, prewarm=True, at_time=at)

    ##############################################################
    ##############################################################
    ##############################################################
