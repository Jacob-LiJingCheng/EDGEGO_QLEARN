import numpy as np
import matplotlib.pyplot as plt
import random
import time
from core.PSTopLayer import ps_top_layer
from core.PSTopLayer_WithoutTwice import ps_top_layer_without_twice  # 可选启用

# 参数配置，对应论文实验
base_parameters = {
    'move_speed': 5,
    'storage_capacity': 100,
    'compute_freq': 8,
    'trans': 2.5,
    'epsilon': 0.001,
    'iteration': 50,
    'distance': None,
    'omega': 0.01
}

max_device_count = 50
device_nums = []
utilization_twice = []
utilization_once = []

for device_count in range(2, max_device_count + 1, 2):
    device_nums.append(device_count)
    parameter_list = base_parameters.copy()
    parameter_list['device_count'] = device_count

    # 生成设备信息（完全对齐 MATLAB）
    device_list = []
    for i in range(device_count):
        device = {
            'num': i + 1,
            'inputsize': random.randint(11, 30),     # randi(20) + 10
            'computation': random.randint(101, 130), # randi(30) + 100
            'storage': random.randint(6, 20),        # randi(15) + 5
            'prior': -1
        }
        device_list.append(device)

    # 随机生成距离矩阵（20~40）
    min_d, max_d = 20, 40
    node_distance = np.zeros((device_count, device_count))
    for i in range(device_count):
        for j in range(i + 1, device_count):
            d = random.randint(min_d, max_d)
            node_distance[i][j] = d
            node_distance[j][i] = d
    parameter_list['distance'] = node_distance

    # 调用主算法（EdgeGo with twice pass）
    ppath, schedule, stay_node_list, completion_time, _, compute_during = ps_top_layer(device_list, parameter_list)

    if completion_time <= 1e-6:
        utilization_twice.append(0)
    else:
        utilization_twice.append(compute_during / completion_time)

    print(f"[{device_count}] Completion: {completion_time:.2f}, Compute: {compute_during:.2f}, Utilization: {utilization_twice[-1]:.4f}")

    # 可选对比算法（Without Twice Pass）
    ppath2, schedule2, stay_node_list2, completionTime2, _, computeDuring2 = ps_top_layer_without_twice(device_list, parameter_list)
    if completionTime2 <= 1e-6:
        utilization_once.append(0)
    else:
        utilization_once.append(computeDuring2 / completionTime2)

# 绘图（从设备数=10开始画）
plt.figure(figsize=(10, 6))
plt.plot(device_nums[4:], utilization_twice[4:], '-d', linewidth=2, label='Go with twice pass')
plt.plot(device_nums[4:], utilization_once[4:], '--o', linewidth=2, label='Go without twice pass')
plt.xlabel('Number of IoT Devices')
plt.ylabel('Utilization of Computation Resource')
plt.ylim([0, 0.8])
plt.grid(True)
plt.legend()
plt.title("EdgeGo: Resource Utilization vs IoT Device Count")
plt.tight_layout()
plt.show()
