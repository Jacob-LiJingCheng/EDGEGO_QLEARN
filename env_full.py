# env_full.py
import numpy as np
import random
from core.PathPlanning import path_planning
from core.ComScheduling import com_scheduling
from core.ComputeDDL import compute_ddl
from core.ComputeNodeCompleteTime import compute_node_complete_time
import math

class EdgeGOFullEnv:
    def __init__(self, device_list, parameter_list, max_steps=20):
        self.device_list = device_list
        self.param = parameter_list.copy()
        self.n = len(device_list)
        self.max_steps = max_steps

        self.reset()

        # 动作空间大小 = noop + rank + 2-opt路径交换
        self.rank_action_count = self.n
        self.swap_action_count = self.n * (self.n - 1) // 2
        self.action_space = 1 + self.rank_action_count + self.swap_action_count

    def reset(self):
        self.schedule = [i + 1 for i in range(self.n)]
        self.ppath = [i + 1 for i in range(self.n) for _ in range(2)]
        self.stay_node_list = []
        self.completion_time, self.compute_during = None, None
        self.completion_time_real, self.compute_during_real = None, None
        self.steps = 0
        return self._get_state()

    def step(self, action):
        done = False
        info = {}

        if action == 0:
            pass  # noop
        elif 1 <= action <= self.rank_action_count:
            node = action - 1
            from core.RankInsert import rank_insert
            self.schedule = rank_insert(self.schedule, node)
        else:
            # 路径交换
            swap_id = action - 1 - self.rank_action_count
            i, j = self._decode_swap(swap_id)
            indices = list(range(0, len(self.ppath), 2))
            visit_order = [self.ppath[k] for k in indices]
            visit_order[i], visit_order[j] = visit_order[j], visit_order[i]
            self.ppath = []
            for dev in visit_order:
                self.ppath.extend([dev, dev])        

        # 模拟当前路径和调度
        ddllist, stoplist = compute_ddl(self.ppath, self.device_list, self.param)
        self.completion_time, _, self.stay_node_list, self.compute_during = compute_node_complete_time_sensitive(
            ddllist, stoplist, self.ppath, self.schedule, 1, self.device_list, self.param
        )

        self.completion_time_real, _, self.stay_node_list_real, self.compute_during_real = compute_node_complete_time(
            ddllist, stoplist, self.ppath, self.schedule, 1, self.device_list, self.param
        )

        # 计算路径长度
        path_len = 0
        for i in range(1, len(self.ppath)):
            u = self.ppath[i - 1] - 1
            v = self.ppath[i] - 1
            path_len += self.param["distance"][u][v]

        # 停留点数量
        stay_count = len(set(self.stay_node_list))

        # 安全处理
        if self.completion_time <= 0:
            reward = -100
        else:
            t   = self.completion_time
            u   = self.compute_during / t
            pl  = path_len / self.n
            sc  = stay_count / self.n

            reward  = - math.log1p(t)          # 时间
            reward += 3.0 * u                  # 利用率
            reward += -0.05 * pl                # 路径长度
            reward += -0.3 * sc                # 停留点
            
            # # 奖励函数（可调系数）
            # alpha = 5.0
            # beta = 0.1
            # gamma = 0.5
            # reward =  (
            #     - math.log1p(self.completion_time) +
            #     alpha * self.compute_during -
            #     beta * path_len -
            #     gamma * stay_count
            # )
            # reward /= 100.0  # 可选归一化

        self.steps += 1
        done = self.steps >= self.max_steps
        return self._get_state(), reward, done, info


    def _get_state(self):
        n = self.n
        # 原有特征：调度与访问次数
        schedule_vec = np.array(self.schedule, dtype=np.float32) / n
        visit_count = np.zeros(n)
        for dev in self.ppath:
            visit_count[dev - 1] += 1
        visit_vec = visit_count / 2.0

        # 存储与时间比例
        storage_now = sum(self.device_list[d - 1]['storage'] for d in set(self.ppath))
        storage_ratio = storage_now / self.param['storage_capacity']
        time_ratio = (self.completion_time or 0) / 100.0

        # 全局特征：平均距离、平均计算、路径总长度
        dist_mat = self.param["distance"]
        all_dist = [dist_mat[i][j] for i in range(n) for j in range(i)]
        avg_dist = np.mean(all_dist) / 50.0
        avg_comp = np.mean([d["computation"] for d in self.device_list]) / 100.0

        # 计算完整路径长度列表
        edge_distances = []
        for i in range(1, len(self.ppath)):
            u = self.ppath[i-1] - 1
            v = self.ppath[i]   - 1
            edge_distances.append(dist_mat[u][v])

        # 取前5条边的距离，padding 0
        k = 5
        if len(edge_distances) >= k:
            local_edges = edge_distances[:k]
        else:
            local_edges = edge_distances + [0] * (k - len(edge_distances))
        local_edges = np.array(local_edges, dtype=np.float32) / 50.0  # 归一化

        # 组合所有特征
        state = np.concatenate([
            schedule_vec,                # n
            visit_vec,                   # n
            [storage_ratio, time_ratio], # 2
            [avg_dist, avg_comp],        # 2
            local_edges                  # k
        ])
        return state



    def _decode_swap(self, idx):
        count = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if count == idx:
                    return i, j
                count += 1
        raise ValueError("Invalid swap index")

    @property
    def action_dim(self):
        return self.action_space
    
    @property
    def state_dim(self):
        return len(self._get_state())
    
    import numpy as np

def compute_node_complete_time_sensitive(ddllist, stoplist, ppath, ranklist, node, devicelist, parameterlist):
    """
    更敏感版本：每个任务初始化就入队，路径顺序显著影响完成时间，仅处理本节点任务。
    """
    computeDuring = 0
    stay_node_list = []

    n = len(devicelist)
    time = 0
    node_finishtime = 0

    # 初始化所有任务（每个为 [device_id, rank, remain_time, finished]）
    tasks = [[i + 1, ranklist[i], devicelist[i]['computation'] / parameterlist['compute_freq'], 0] for i in range(n)]
    tasks.sort(key=lambda x: x[1])  # 按 rank 升序排序

    # 找 node 在 ppath 中的第二次出现位置
    node_indices = [i for i, x in enumerate(ppath) if x == node]
    if len(node_indices) < 2:
        raise ValueError("节点在路径中未出现两次")
    nodep = node_indices[-1]

    # 模拟路径移动与任务处理
    for p in range(2 * n - 1):
        current = ppath[p]
        next_node = ppath[p + 1]

        if p == nodep:
            node_finishtime = time

        staytime = stoplist[p]
        movetime = parameterlist['distance'][current - 1][next_node - 1] / parameterlist['move_speed']
        computetime = staytime

        # 查找当前节点任务是否未完成
        task_idx = next((i for i, t in enumerate(tasks) if t[0] == current and t[3] == 0), None)

        if task_idx is not None:
            task = tasks[task_idx]
            if computetime >= task[2]:
                computeDuring += task[2]
                computetime -= task[2]
                tasks[task_idx][2] = 0
                tasks[task_idx][3] = 1
            else:
                computeDuring += computetime
                tasks[task_idx][2] -= computetime
                computetime = 0

        time += staytime + movetime

    total_finishtime = time
    if nodep == 2 * n - 1:
        node_finishtime = time

    return total_finishtime, node_finishtime, stay_node_list, computeDuring


