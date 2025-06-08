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

def experiment_iotscale():
    # --- 读取 DQN 超参数 ---
    cfg = yaml.safe_load(open("config.yml", "r", encoding="utf-8"))
    dqn_cfg = cfg["dqn"]

    # --- 环境固定参数（基于原论文设定） ---
    move_speed = 5                  # 移动速度
    storage_capacity = 100          # 存储容量
    compute_freq = 8                # 计算频率
    trans = 2.5                     # 传输速率
    epsilon = 0.001                 # ε-贪心下界
    iteration = 50                  # 最大迭代次数
    omega = 0.01                    # 权重参数

    # --- 实验设置 ---
    max_device = 50
    step = 2
    device_counts = list(range(2, max_device + 1, step))
    repeat_times = 20  # 每点重复实验次数

    # --- 数据结构 ---
    util_iptu = defaultdict(list)
    util_dqn = defaultdict(list)

    # --- 重复实验 ---
    for n in device_counts:
        print(f"\n=== Device Count = {n} ===")
        for run in range(repeat_times):
            # 生成设备列表
            device_list = []
            for i in range(n):
                device_list.append({
                    'id': i + 1,
                    'inputsize': random.randint(11, 30),
                    'computation': random.randint(101, 130),
                    'storage': random.randint(6, 20)
                })

            # 构造随机距离矩阵
            min_d, max_d = 20, 40
            dist = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    d = random.randint(min_d, max_d)
                    dist[i][j] = dist[j][i] = d

            # 环境参数
            param = {
                'device_count': n,
                'move_speed': move_speed,
                'trans': trans,
                'compute_freq': compute_freq,
                'distance': dist.tolist(),
                'storage_capacity': storage_capacity,
                'epsilon': epsilon,
                'iteration': iteration,
                'omega': omega
            }

            # === IPTU 方法 ===
            _, _, _, completion_time, _, compute_during = ps_top_layer(device_list, param)
            util1 = compute_during / completion_time if completion_time > 0 else 0
            util_iptu[n].append(util1)

            # === DQN 方法 ===
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
            util2 = env.compute_during_real / env.completion_time_real if env.completion_time > 0 else 0
            util_dqn[n].append(util2)

            print(f" Run {run+1}/{repeat_times}: IPTU={util1:.4f}, DQN={util2:.4f}")

    # --- 配对 t 检验 ---
    print("\n=== Paired t-test Results ===")
    for n in device_counts:
        t_stat, p_val = ttest_rel(util_dqn[n], util_iptu[n])
        sig = '✅' if p_val < 0.05 else '—'
        print(f"Devices={n} | t={t_stat:.4f}, p={p_val:.4f} {sig}")

    # --- 可视化 Mean±Std ---设备数>=10的图
    filtered_counts = [n for n in device_counts if n >= 10]
    means_iptu = [np.mean(util_iptu[n]) for n in filtered_counts]
    means_dqn = [np.mean(util_dqn[n]) for n in filtered_counts]
    std_iptu = [np.std(util_iptu[n]) for n in filtered_counts]
    std_dqn = [np.std(util_dqn[n]) for n in filtered_counts]

    plt.figure(figsize=(8,5))
    plt.errorbar(filtered_counts, means_iptu, yerr=std_iptu, fmt='-o', label='IPTU', capsize=3)
    plt.errorbar(filtered_counts, means_dqn, yerr=std_dqn, fmt='--d', label='DQN', capsize=3)
    plt.xlabel('Number of IoT Devices')
    plt.ylabel('Utilization')
    plt.title('Utilization vs IoT Scale (Mean±Std)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('results/iotscale_utilization_stats.png', dpi=150)
    plt.show()

    # --- 保存数据 ---
    np.save('results/util_iotscale_iptu.npy', dict(util_iptu))
    np.save('results/util_iotscale_dqn.npy', dict(util_dqn))
    with open('results/utilization_iotscale_comparison.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['DeviceCount', 'Run', 'IPTU', 'DQN'])
        for n in device_counts:
            for i in range(repeat_times):
                writer.writerow([n, i+1, util_iptu[n][i], util_dqn[n][i]])

if __name__ == '__main__':
    experiment_iotscale()
