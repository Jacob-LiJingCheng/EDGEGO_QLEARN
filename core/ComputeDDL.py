import numpy as np

def compute_ddl(ppath, device_list, parameter_list):
    """
    根据路径计算每个节点的DDL（任务上传后返回的时间）。
    参数:
        ppath: List[int]，移动路径，长度为 2 * 设备数
        device_list: List[dict]，每个字典表示一个设备，包含 'inputsize' 字段
        parameter_list: dict，包含 'trans'（上传速率），'compute_freq'（计算频率），
                        'distance'（距离矩阵，二维数组），'move_speed'（移动速度）
    返回:
        ddl_list: np.ndarray，任务完成时间列表
        stop_list: np.ndarray，每次停留的时长列表
    """

    n_devices = len(device_list)
    ddl_list = np.zeros(n_devices)
    stop_list = np.zeros(2 * n_devices)

    i = 0
    while i < 2 * n_devices - 1:
        current = ppath[i]
        next_node = ppath[i + 1]
        if current == next_node:
            # 直接停留原地计算
            ddl_list[current - 1] = -1  # MATLAB 是 1-indexed
            stop_list[i] = device_list[current - 1]['inputsize'] / parameter_list['trans']
            stop_list[i + 1] = device_list[current - 1]['inputsize'] / parameter_list['compute_freq']
        else:
            # 二次卸载，需要计算第一次上传时间
            is_first = True
            for j in range(i):
                if ppath[j] == current:
                    is_first = False
                    break
            if is_first:
                stop_list[i] = device_list[current - 1]['inputsize'] / parameter_list['trans']
        i += 1

    # 计算二次卸载节点的DDL
    for i in range(n_devices):
        if ddl_list[i] == -1:
            continue
        indices = [idx for idx, val in enumerate(ppath) if val == i + 1]
        if len(indices) < 2:
            continue  # 路径中出现次数不足
        pfirst, psecond = indices[:2]

        total_distance = 0
        total_stoptime = 0
        for j in range(pfirst, psecond):
            total_distance += parameter_list['distance'][ppath[j] - 1][ppath[j + 1] - 1]
            total_stoptime += stop_list[j + 1]
        ddl_list[i] = total_stoptime + total_distance / parameter_list['move_speed']

    return ddl_list, stop_list
