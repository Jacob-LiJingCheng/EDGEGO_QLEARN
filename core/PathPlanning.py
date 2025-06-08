import random
import copy
from .ComputeDDL import compute_ddl
from .ComputeNodeCompleteTime import compute_node_complete_time

def path_planning(last_completion_time, last_path, schedule, stay_node_list, devicelist, parameterlist):
    temp_path = last_path.copy()
    min_completion_time = float('inf')  # 强制从头开始接受第一条可行路径
    min_path = temp_path.copy()
    min_stay_node_list = stay_node_list.copy()
    compute_during = 0.0

    n = len(temp_path)
    for _ in range(parameterlist['iteration']):
        # 2-OPT 翻转
        opt_i, opt_j = sorted(random.sample(range(n), 2))
        if opt_j - opt_i >= 1:
            temp_path[opt_i:opt_j+1] = temp_path[opt_i:opt_j+1][::-1]

        # 存储限制判断
        task_queue = []
        visit_count = {}
        current_storage = 0
        flag_storage = True

        for point in temp_path:
            visit_count[point] = visit_count.get(point, 0) + 1
            if visit_count[point] == 1:
                task_queue.append(point)
                current_storage = sum(devicelist[i-1]['storage'] for i in task_queue)
                if current_storage > parameterlist['storage_capacity']:
                    flag_storage = False
                    break
            elif visit_count[point] == 2:
                if point in task_queue:
                    task_queue.remove(point)

        if flag_storage:
            # 满足约束，计算新路径代价
            temp_ddllist, temp_stoplist = compute_ddl(temp_path, devicelist, parameterlist)
            temp_completion_time, _, temp_stay_node_list, temp_compute_time = compute_node_complete_time(
                temp_ddllist, temp_stoplist, temp_path, schedule, 1, devicelist, parameterlist
            )

            temp_completion_time = max(temp_completion_time, 1e-6)
            temp_compute_time = max(temp_compute_time, 1e-6)

            # ✅ 修复：第一次合法路径必须接受
            if min_completion_time == float('inf') or (
                temp_completion_time < min_completion_time and
                abs(temp_completion_time - min_completion_time) > parameterlist['epsilon']
            ):
                min_completion_time = temp_completion_time
                min_path = temp_path.copy()
                min_stay_node_list = temp_stay_node_list.copy()
                compute_during = temp_compute_time

    return min_path, min_stay_node_list, min_completion_time, compute_during
