import numpy as np
import random
import yaml
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
from scipy.stats import ttest_rel

from core.PSTopLayer import ps_top_layer
from env_full import EdgeGOFullEnv
from dqn import DQNAgent

def experiment_distance():
    # --- 读取配置 ---
    cfg = yaml.safe_load(open("config.yml", "r", encoding="utf-8"))
    dqn_cfg = cfg["dqn"]

    # --- 环境固定参数（基于原论文设定） ---
    device_count = 10               # 设备数量
    move_speed = 5                  # 移动速度
    storage_capacity = 100          # 存储容量
    compute_freq = 8                # 计算频率
    trans = 2.5                     # 传输速率
    epsilon = 1e-5                  # ε-贪心下界
    iteration = 30                  # 最大迭代次数
    omega = 1e-4                    # 权重参数

    # --- 固定随机生成一次设备配置 ---
    min_input, max_input = 11, 30
    min_comp, max_comp = 101, 120
    min_storage, max_storage = 6, 20
    device_list = []
    for i in range(device_count):
        device_list.append({
            'id': i+1,
            'inputsize': random.randint(min_input, max_input),
            'computation': random.randint(min_comp, max_comp),
            'storage': random.randint(min_storage, max_storage)
        })

    # --- 实验参数 ---
    distances = list(range(1, 251, 5))
    repeat_times = 20  # 每点重复实验次数

    # --- 数据结构 ---
    util_iptu = defaultdict(list)
    util_dqn = defaultdict(list)

    # --- 重复实验 ---
    for dist_avg in distances:
        print(f"\n=== Distance={dist_avg} ===")
        # 构造随机扰动的距离矩阵（平均为 dist_avg）
        node_distance = [[0]*device_count for _ in range(device_count)]
        for i in range(device_count):
            for j in range(i):
                d = int(random.gauss(mu=dist_avg, sigma=dist_avg * 0.3))
                d = max(1, min(50, d))
                node_distance[i][j] = node_distance[j][i] = d


        param = {
            'device_count': device_count,
            'move_speed': move_speed,
            'trans': trans,
            'compute_freq': compute_freq,
            'distance': node_distance,
            'storage_capacity': storage_capacity,
            'epsilon': epsilon,
            'iteration': iteration,
            'omega': omega
        }

        for run in range(repeat_times):
            # IPTU
            _, _, _, completion_time_iptu, _, compute_during_iptu = ps_top_layer(device_list, param)
            util1 = compute_during_iptu / completion_time_iptu if completion_time_iptu > 0 else 0
            util_iptu[dist_avg].append(util1)

            # DQN
            env = EdgeGOFullEnv(device_list, param)
            agent = DQNAgent(env.state_dim, env.action_dim, dqn_cfg)
            for ep in range(dqn_cfg['episodes']):
                s, done = env.reset(), False
                while not done:
                    a = agent.select_action(s)
                    s2, r, done, _ = env.step(a)
                    agent.push_transition(s, a, r, s2, float(done))
                    agent.update()
                    s = s2
            compute_during_dqn = env.compute_during_real
            completion_time_dqn = env.completion_time_real
            util2 = compute_during_dqn / completion_time_dqn if completion_time_dqn > 0 else 0
            util_dqn[dist_avg].append(util2)

            print(f" Run {run+1}/{repeat_times}: "
                  f"IPTU={util1:.4f}, DQN={util2:.4f} ")


    # --- 配对 t 检验 ---
    print("\n=== Paired t-test Results ===")
    for dist_avg in distances:
        t_stat, p_val = ttest_rel(util_dqn[dist_avg], util_iptu[dist_avg])
        sig = '✅' if p_val < 0.05 else '—'
        print(f"Distance={dist_avg} | t={t_stat:.4f}, p={p_val:.4f} {sig}")

    # --- 可视化 Mean±Std ---
    means_iptu = [np.mean(util_iptu[d]) for d in distances]
    means_dqn = [np.mean(util_dqn[d]) for d in distances]
    std_iptu = [np.std(util_iptu[d]) for d in distances]
    std_dqn = [np.std(util_dqn[d]) for d in distances]

    plt.figure(figsize=(8,5))
    plt.errorbar(distances, means_iptu, yerr=std_iptu, fmt='-o', label='IPTU', capsize=3)
    plt.errorbar(distances, means_dqn, yerr=std_dqn, fmt='--d', label='DQN', capsize=3)
    plt.xlabel('Average Distance')
    plt.ylabel('Utilization')
    plt.title('Utilization vs Distance (Mean±Std)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/distance_utilization_stats.png', dpi=150)
    plt.show()

    # --- 保存数据 ---
    np.save('results/util_distance_iptu.npy', dict(util_iptu))
    np.save('results/util_distance_dqn.npy', dict(util_dqn))
    with open('results/utilization_distance_comparison.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Distance', 'Run', 'IPTU', 'DQN'])
        for d in distances:
            for i in range(repeat_times):
                writer.writerow([d, i+1, util_iptu[d][i], util_dqn[d][i]])

if __name__ == '__main__':
    experiment_distance()
