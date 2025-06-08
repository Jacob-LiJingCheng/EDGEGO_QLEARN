def com_scheduling(last_time, last_schedule, ppath, device_list, parameter_list):
    """
    根据固定路径 ppath 调整任务调度 schedule，使每个任务尽量在其 DDL 之前完成。

    参数：
        last_time: float，上一次任务调度的完成时间
        last_schedule: List[int]，初始任务优先级列表（rank越小优先级越高）
        ppath: List[int]，服务器移动路径
        device_list: List[dict]，每个设备含 'inputsize' 和 'computation'
        parameter_list: dict，包括 'distance', 'trans', 'compute_freq', 'move_speed'

    返回：
        schedule: List[int]，新的调度优先级列表
        stay_node_list: List[Tuple[int, float]]，服务器逗留时间记录
        completionTime: float，总任务完成时间
    """
    import copy

    from .ComputeDDL import compute_ddl
    from .ComputeNodeCompleteTime import compute_node_complete_time
    from .RankInsert import rank_insert

    total_comptime = last_time
    ddl_list, temp_stop = compute_ddl(ppath, device_list, parameter_list)
    schedule = copy.deepcopy(last_schedule)
    stay_node_list = []

    for i in range(len(device_list)):
        if ddl_list[i] == -1:
            continue  # 本地计算节点，无需更改优先级

        total_comptime, node_comptime, temp_stay_list, _ = compute_node_complete_time(
            ddl_list, temp_stop, ppath, schedule, i + 1, device_list, parameter_list
        )
        stay_node_list = temp_stay_list

        # 调高优先级直到满足 DDL 或优先级无法再提升
        while node_comptime > ddl_list[i] and schedule[i] > 1:
            schedule = rank_insert(schedule, i)
            total_comptime, node_comptime, temp_stay_list, _ = compute_node_complete_time(
                ddl_list, temp_stop, ppath, schedule, i + 1, device_list, parameter_list
            )
            stay_node_list = temp_stay_list

    completionTime = total_comptime

    if last_time < total_comptime:
        # 本次调度时间不优于上次，保留原始调度
        return last_schedule, stay_node_list, last_time
    else:
        return schedule, stay_node_list, completionTime
