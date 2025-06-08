"""
core/PSTopLayer_WithoutTwice.py

顶层控制器（对比算法）：
 - 固定 path → 调度  : com_scheduling
 - 固定调度 → path   : path_planning_without_twice
不允许二次卸载 ⇒ 服务器必须等任务算完才能离开
"""

import random
import math

from .ComScheduling import com_scheduling
from .PathPlanning_WithoutTwice import path_planning_without_twice


def ps_top_layer_without_twice(device_list, parameter_list):
    """
    :return: (ppath, schedule, stay_node_list, completion_time,
              completion_time_hist, compute_during)
    """
    n = len(device_list)

    # 初始 path: [1,1,2,2,...]
    ppath = [i + 1 for i in range(n) for _ in range(2)]
    schedule = [i + 1 for i in range(n)]      # rank 1,2,3...
    stay_node_list = []
    completion_time = float('inf')
    compute_during = 0.0

    last_ppath = ppath[:]
    last_schedule = schedule[:]
    last_stay_list = stay_node_list[:]
    last_completion = completion_time
    last_compute = 0.0
    history = []

    for _ in range(parameter_list['iteration']):
        # ---------- 固定 path, 优化调度 ----------
        schedule, stay_node_list, completion_time = com_scheduling(
            last_completion, last_schedule, last_ppath,
            device_list, parameter_list
        )

        # ---------- 固定调度, 优化 path ----------
        ppath, stay_node_list, completion_time, compute_during = \
            path_planning_without_twice(
                last_completion, last_ppath, schedule,
                stay_node_list, device_list, parameter_list
            )

        # ---------- 模拟退火式接受 ----------
        delta = completion_time - last_completion
        prob = 1 / (1 + math.exp(delta / parameter_list['omega']))
        if random.random() <= prob:
            # 接受新解
            last_ppath = ppath[:]
            last_schedule = schedule[:]
            last_stay_list = stay_node_list[:]
            last_completion = completion_time
            last_compute = compute_during
        history.append(last_completion)

        # 下一轮以“当前最优解”继续搜索
        ppath = last_ppath[:]
        schedule = last_schedule[:]
        stay_node_list = last_stay_list[:]
        completion_time = last_completion
        compute_during = last_compute

    return (last_ppath, last_schedule, last_stay_list,
            last_completion, history, last_compute)
