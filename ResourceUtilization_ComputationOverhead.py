import numpy as np
import matplotlib.pyplot as plt
import random

from core.PSTopLayer import ps_top_layer
from core.PSTopLayer_WithoutTwice import ps_top_layer_without_twice

def resource_utilization_computation_overhead():
    parameter_list = {
        'device_count': 10,
        'move_speed': 5,
        'storage_capacity': 100,
        'compute_freq': 8,
        'trans': 2.5,
        'epsilon': 1e-5,
        'iteration': 50,
        'omega': 1e-4,
        'distance': None
    }

    max_computation = 120
    computation_step = 5
    computation_num = []
    utilization_twice = []
    utilization_once = []

    device_count = parameter_list['device_count']
    min_distance = 20
    max_distance = 50

    # 生成对称距离矩阵
    node_distance = np.zeros((device_count, device_count))
    for i in range(device_count):
        for j in range(i + 1, device_count):
            d = random.randint(min_distance, max_distance)
            node_distance[i][j] = node_distance[j][i] = d
    parameter_list['distance'] = node_distance

    for num in range(0, max_computation + 1, computation_step):
        computation_num.append(num)
        device_list = []
        for i in range(device_count):
            device = {
                'num': i + 1,
                'inputsize': 30,  # 20 + 10
                'computation': 10 + num,
                'storage': 20     # 15 + 5
            }
            device_list.append(device)

        # EdgeGO: with twice-pass strategy
        ppath, schedule, stay_node_list, completion_time, completion_time_iter, compute_during = ps_top_layer(
            device_list, parameter_list
        )

        if completion_time == 0:
            utilization_twice.append(0)
        else:
            utilization_twice.append(device_count / completion_time)

        # 对比算法: 无二次卸载策略（如需启用取消注释）
        ppath2, schedule2, stay_node_list2, completion_time2, _, _ = ps_top_layer_without_twice(
            device_list, parameter_list
        )
        if completion_time2 == 0:
            utilization_once.append(0)
        else:
            utilization_once.append(device_count / completion_time2)

    # 绘图
    plt.plot(computation_num, utilization_twice, '-d', label='Go with twice pass', linewidth=2)
    plt.plot(computation_num, utilization_once, '--o', label='Go without twice pass', linewidth=2)
    plt.xlabel('Average Computation Overhead')
    plt.ylabel('Utilization of Computation Resource')
    plt.legend()
    plt.grid(True)
    plt.title('Computation Resource Utilization vs Overhead')
    plt.show()

if __name__ == '__main__':
    resource_utilization_computation_overhead()