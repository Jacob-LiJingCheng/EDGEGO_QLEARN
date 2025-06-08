import random
import copy

def path_planning_without_twice(last_completion_time, last_path, schedule, stay_node_list, devicelist, parameterlist):
    """
    固定任务优先级 schedule，路径中每个节点必须等待任务在本地完全计算完成后再移动。
    无需考虑存储约束。

    参数：
        last_completion_time: float，上次路径的任务完成时间
        last_path: List[int]，上次路径（每个设备成对出现 [i, i]）
        schedule: List[int]，任务优先级（未使用）
        stay_node_list: List[Tuple[int, float]]，保留字段
        devicelist: List[dict]，包含 'inputsize', 'computation'
        parameterlist: dict，包括：
            - 'distance': 2D array，节点间距离
            - 'trans': float，上传速率
            - 'compute_freq': float，计算频率
            - 'move_speed': float，移动速度
            - 'iteration': int，迭代次数
            - 'epsilon': float，优化容忍阈值

    返回：
        ppath: List[int]，优化后的路径
        stay_node_list: 原始（无变更）
        completion_time: 最短总时间
        compute_during: 总计算耗时
    """

    n = len(devicelist)
    temp_path = last_path.copy()
    temp_completion_time = last_completion_time
    temp_compute_during = 0
    min_completion_time = temp_completion_time
    min_path = temp_path.copy()
    min_stay_node_list = stay_node_list.copy()

    # 初始化设备顺序，只使用下标为偶数位置的元素 [1,1,2,2,...] -> [1,2,...]
    temp_list = [temp_path[2*i + 1] for i in range(n)]

    for _ in range(parameterlist['iteration']):
        # 2-OPT 翻转 temp_list 中的设备顺序
        opt_i, opt_j = sorted(random.sample(range(n), 2))
        opt_n = opt_j - opt_i + 1
        left = opt_i
        right = opt_j

        while left < right:
            temp_list[left], temp_list[right] = temp_list[right], temp_list[left]
            left += 1
            right -= 1

        # 构建新的路径（无二次卸载策略：每个节点成对出现）
        temp_path = []
        for device in temp_list:
            temp_path.extend([device, device])  # [1,1,3,3,...]

        # 计算该路径下的完成时间
        temp_completion_time = 0
        temp_compute_during = 0

        for p in range(len(temp_path)):
            device_id = temp_path[p] - 1
            if p % 2 == 0:
                # 上传
                upload_time = devicelist[device_id]['inputsize'] / parameterlist['trans']
                temp_completion_time += upload_time
            else:
                # 计算 + 移动（若非末尾）
                compute_time = devicelist[device_id]['computation'] / parameterlist['compute_freq']
                if p != len(temp_path) - 1:
                    move_time = parameterlist['distance'][temp_path[p]-1][temp_path[p+1]-1] / parameterlist['move_speed']
                else:
                    move_time = 0
                temp_completion_time += compute_time + move_time
                temp_compute_during += compute_time

        if (temp_completion_time < min_completion_time and
            abs(temp_completion_time - min_completion_time) > parameterlist['epsilon']):
            min_completion_time = temp_completion_time
            min_path = temp_path.copy()
            min_stay_node_list = stay_node_list.copy()

    return min_path, min_stay_node_list, min_completion_time, temp_compute_during
