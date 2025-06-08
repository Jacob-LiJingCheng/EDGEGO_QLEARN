def rank_insert(ranklist, node):
    """
    将节点 node 的 rank 提高一位（与前一位节点交换位置）
    
    参数：
        ranklist: List[int]，长度为 n，ranklist[i] 表示节点 i 的优先级（rank 越小优先级越高）
        node: int，要提升的节点索引（从 0 开始）

    返回：
        ranklist: List[int]，更新后的 rank 列表
    """
    # 当前节点的 rank 值
    node_rank = ranklist[node]
    
    if node_rank <= 1:
        return ranklist  # 已是最高优先级，无法再提升

    try:
        # 找到当前 rank - 1 的节点索引
        q_index = ranklist.index(node_rank - 1)

        # 交换两者 rank
        ranklist[node] = node_rank - 1
        ranklist[q_index] = node_rank
    except ValueError:
        pass  # 没有 rank-1 的节点，不做变动

    return ranklist
