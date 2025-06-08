import random
import math
import copy

from .ComScheduling import com_scheduling
from .PathPlanning import path_planning

def ps_top_layer(device_list, parameter_list, debug=False):
    device_count = len(device_list)

    # 1. 随机初始化路径 [i,i,...]
    shuffled = random.sample(range(1, device_count + 1), device_count)
    ppath = [i for val in shuffled for i in (val, val)]  # 打乱后的[1,1,2,2,...]
    schedule = [i + 1 for i in range(device_count)]

    completion_time = float('inf')
    compute_during = 0.0
    stay_node_list = []

    last_ppath = copy.deepcopy(ppath)
    last_schedule = copy.deepcopy(schedule)
    last_stay_node_list = copy.deepcopy(stay_node_list)
    last_completion_time = completion_time
    last_compute_during = 0.0

    completion_time_iter = []

    for it in range(parameter_list['iteration']):
        # --- 固定路径优化调度 ---
        schedule, stay_node_list, completion_time = com_scheduling(
            last_completion_time, last_schedule, last_ppath, device_list, parameter_list
        )

        # --- 固定调度优化路径 ---
        ppath, stay_node_list, completion_time, compute_during = path_planning(
            last_completion_time, last_ppath, schedule, stay_node_list, device_list, parameter_list
        )

        # --- 安全下限保护 ---
        completion_time = max(completion_time, 1e-6)
        compute_during = max(compute_during, 1e-6)

        # --- 模拟退火接受概率，加入最小下限 ---
        delta = completion_time - last_completion_time
        omega = parameter_list.get('omega', 0.0001)
        if omega != 0:
            try:
                exp_arg = min(delta / omega, 700)
                prob = max(1e-3, 1 / (1 + math.exp(exp_arg)))
            except OverflowError:
                prob = 1e-3
        else:
            prob = 0.0

        dice = random.random()

        if debug:
            print(f"[Iter {it}] Δ={delta:.6f}, prob={prob:.6f}, dice={dice:.6f}, completion={completion_time:.2f}, compute={compute_during:.2f}")

        # --- 接受/拒绝更新 ---
        if dice <= prob:
            last_ppath = copy.deepcopy(ppath)
            last_schedule = copy.deepcopy(schedule)
            last_stay_node_list = copy.deepcopy(stay_node_list)
            last_completion_time = completion_time
            last_compute_during = compute_during
            completion_time_iter.append(completion_time)
        else:
            completion_time_iter.append(last_completion_time)

        # --- 回退为当前最优 ---
        ppath = copy.deepcopy(last_ppath)
        schedule = copy.deepcopy(last_schedule)
        stay_node_list = copy.deepcopy(last_stay_node_list)
        completion_time = last_completion_time
        compute_during = last_compute_during

    return ppath, schedule, stay_node_list, completion_time, completion_time_iter, compute_during
