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

def experiment_computation_overhead():
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
    iteration = 50                  # 最大迭代次数
    omega = 1e-4                    # 权重参数

    # --- 生成一次对称距离矩阵 ---
    min_dist, max_dist = 20, 50
    distance = np.zeros((device_count, device_count))
    for i in range(device_count):
        for j in range(i+1, device_count):
            d = random.randint(min_dist, max_dist)
            distance[i][j] = distance[j][i] = d

    # --- 实验设置 ---
    base_comp = 10
    max_overhead = 120
    step = 5
    overheads = list(range(0, max_overhead+1, step))
    repeat_times = 20  # 每点重复实验次数

    # --- 数据结构 ---
    util_iptu = defaultdict(list)
    util_dqn = defaultdict(list)

    # --- 重复实验 ---
    for extra in overheads:
        comp_val = base_comp + extra
        print(f"\n=== Overhead={comp_val} ===")

        # 设备列表（固定）
        device_list = [
            {'id': i+1, 'inputsize': 30, 'computation': comp_val, 'storage': 20}
            for i in range(device_count)
        ]

        param = {
            'device_count': device_count,
            'move_speed': move_speed,
            'trans': trans,
            'compute_freq': compute_freq,
            'distance': distance.tolist(),
            'storage_capacity': storage_capacity,
            'epsilon': epsilon,
            'iteration': iteration,
            'omega': omega
        }

        for run in range(repeat_times):
            # IPTU
            _, _, _, completion_time, _, compute_during = ps_top_layer(device_list, param)
            util1 = device_count / completion_time if completion_time > 0 else 0
            util_iptu[extra].append(util1)

            # DQN
            env = EdgeGOFullEnv(device_list, param)
            agent = DQNAgent(env.state_dim, env.action_dim, dqn_cfg)
            for ep in range(dqn_cfg['episodes']):
                state, done = env.reset(), False
                while not done:
                    action = agent.select_action(state)
                    next_state, reward, done, _ = env.step(action)
                    agent.push_transition(state, action, reward, next_state, float(done))
                    agent.update()
                    state = next_state
            util2 = device_count / env.completion_time_real if env.completion_time > 0 else 0
            util_dqn[extra].append(util2)
            print(f" Run {run+1}/{repeat_times}: IPTU={util1:.4f}, DQN={util2:.4f}")

    # --- Paired t-test ---
    print("\n=== Paired t-test Results ===")
    for extra in overheads:
        t_stat, p_val = ttest_rel(util_dqn[extra], util_iptu[extra])
        sig = '✅' if p_val < 0.05 else '—'
        print(f"Overhead={base_comp+extra} | t={t_stat:.4f}, p={p_val:.4f} {sig}")

    # --- 可视化 Mean±Std ---
    means_iptu = [np.mean(util_iptu[o]) for o in overheads]
    means_dqn = [np.mean(util_dqn[o]) for o in overheads]
    std_iptu = [np.std(util_iptu[o]) for o in overheads]
    std_dqn = [np.std(util_dqn[o]) for o in overheads]

    plt.figure(figsize=(8,5))
    plt.errorbar(overheads, means_iptu, yerr=std_iptu, fmt='-o', label='IPTU', capsize=3)
    plt.errorbar(overheads, means_dqn, yerr=std_dqn, fmt='--d', label='DQN', capsize=3)
    plt.xlabel('Extra Computation Overhead')
    plt.ylabel('Utilization')
    plt.title('Utilization vs Computation Overhead (Mean±Std)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/computation_overhead_stats.png', dpi=150)
    plt.show()

    # --- 保存数据 ---
    np.save('results/util_overhead_iptu.npy', dict(util_iptu))
    np.save('results/util_overhead_dqn.npy', dict(util_dqn))
    with open('results/utilization__overhead_comparison.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Overhead', 'Run', 'IPTU', 'DQN'])
        for o in overheads:
            for i in range(repeat_times):
                writer.writerow([base_comp+o, i+1, util_iptu[o][i], util_dqn[o][i]])

if __name__ == '__main__':
    experiment_computation_overhead()