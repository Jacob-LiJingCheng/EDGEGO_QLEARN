from collections import deque
import numpy as np
import random, torch

# 经验回放模块
class ReplayBuffer:
    def __init__(self, capacity, batch_size=64):
        self.buf = deque(maxlen=capacity)
        self.batch_size = batch_size

    # 添加一条状态转移记录
    def push(self, *transition):
        self.buf.append(tuple(transition))

    # 从经验池中随机采样一批数据
    # |元素名| 说明                  
    # | ---- | ------------------- 
    # | `s`  | 当前状态 (state)        
    # | `a`  | 所采取的动作 (action)     
    # | `r`  | 得到的奖励 (reward)      
    # | `s2` | 下一个状态 (next\_state) 
    # | `d`  | 是否终止 (done)         

    def sample(self):
        batch = random.sample(self.buf, self.batch_size)
        s, a, r, s2, d = zip(*batch)
        # 1) 先把 list of arrays 变成一个 np.ndarray
        s  = np.stack(s)   # shape: (batch, state_dim)
        s2 = np.stack(s2)
        # a,r,d 已经是标量或一维列表，给它们做一个简单的堆叠
        a  = np.array(a)
        r  = np.array(r, dtype=np.float32)
        d  = np.array(d, dtype=np.float32)

        # 2) 再一起一次性转换为 Tensor
        return (
            torch.from_numpy(s).float(),
            torch.from_numpy(a).long().unsqueeze(1),
            torch.from_numpy(r).float().unsqueeze(1),
            torch.from_numpy(s2).float(),
            torch.from_numpy(d).float().unsqueeze(1),
        )
    
    # 经验池中的数据量
    def __len__(self):
        return len(self.buf)
