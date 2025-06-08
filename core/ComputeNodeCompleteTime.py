import numpy as np

def compute_node_complete_time(ddllist, stoplist, ppath, ranklist, node, devicelist, parameterlist):
    """
    模拟节点 node 完成时间的计算过程。
    
    参数：
        ddllist: np.ndarray，DDL 列表
        stoplist: np.ndarray，停留时间列表
        ppath: List[int]，移动路径
        ranklist: List[int]，任务优先级
        node: int，要分析的节点编号（从 1 开始）
        devicelist: List[dict]，每个设备为一个 dict，包含 'computation'
        parameterlist: dict，包含 'distance'（二维数组），'move_speed'，'compute_freq'

    返回：
        total_finishtime: float，总完成时间
        node_finishtime: float，节点 node 完成时间
        stay_node_list: List[Tuple[int, float]]，逗留节点及时间
        computeDuring: float，累计计算时间
    """

    computeDuring = 0
    tasks = []  # 每个任务为 [device_id, rank, remaining_compute_time, is_finished]
    stay_node_list = []

    time = 0
    node_finishtime = 0
    n = len(devicelist)
    node_indices = [i for i, x in enumerate(ppath) if x == node]
    if len(node_indices) < 2:
        raise ValueError("节点在路径中未出现两次")
    nodep = node_indices[-1]  # 第二次出现的位置

    p = 0
    while p < 2 * n - 1:
        current = ppath[p]
        next_node = ppath[p + 1]

        if p == nodep:
            node_finishtime = time

        staytime = stoplist[p]
        movetime = parameterlist['distance'][current - 1][next_node - 1] / parameterlist['move_speed']
        if ddllist[current - 1] != -1:
            computetime = staytime + movetime
        else:
            computetime = movetime

        surplustime = 0

        # 检查当前节点是否已在任务队列中
        task_ids = [t[0] for t in tasks]
        if current not in task_ids:
            # 第一次访问，入队
            comp_time = devicelist[current - 1]['computation'] / parameterlist['compute_freq']
            tasks.append([current, ranklist[current - 1], comp_time, 0])
            tasks.sort(key=lambda x: x[1])  # 按 rank 升序
        else:
            # 按 rank 排序
            tasks.sort(key=lambda x: x[1])
            i = 0
            while i < len(tasks):
                if computetime >= tasks[i][2]:
                    computetime -= tasks[i][2]
                    computeDuring += tasks[i][2]
                    tasks[i][2] = 0
                    tasks[i][3] = 1
                    i += 1
                else:
                    computeDuring += computetime
                    tasks[i][2] -= computetime
                    computetime = 0
                    break

            # 第二次访问，判断是否需要原地完成剩余任务
            for idx, t in enumerate(tasks):
                if t[0] == current:
                    if t[3] == 1:
                        tasks.pop(idx)
                    else:
                        # 处理优先级更高任务
                        for j in range(idx + 1):
                            if tasks[j][3] == 0:
                                surplustime += tasks[j][2]
                                computeDuring += tasks[j][2]
                                tasks[j][2] = 0
                                tasks[j][3] = 1
                        tasks.pop(idx)
                        stay_node_list.append((current, surplustime))
                    break

        time += staytime + movetime + surplustime
        p += 1

    total_finishtime = time
    if nodep == 2 * n - 1:
        node_finishtime = time

    return total_finishtime, node_finishtime, stay_node_list, computeDuring
