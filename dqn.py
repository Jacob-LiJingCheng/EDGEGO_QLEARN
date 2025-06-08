import torch
import torch.nn as nn
import torch.optim as optim
import random
from utils.replay import ReplayBuffer

# 定义一个三层的前馈神经网络（Fully Connected NN）
class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 当前网络 qnet: 训练中的 Q 网络
        # 目标网络 target: 仅用于计算目标值，每隔一段时间从 qnet 同步一次
        self.qnet   = QNet(state_dim, action_dim).to(self.device)
        self.target = QNet(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.qnet.state_dict())
        # 引入经验回放缓冲池
        self.buffer = ReplayBuffer(capacity=cfg["buffer"], batch_size=cfg["batch"])
        # 使用 Adam 优化器训练 qnet
        self.opt    = optim.Adam(self.qnet.parameters(), lr=cfg["lr"])
        self.gamma  = cfg["gamma"]
        self.epsilon = cfg["eps_start"]
        self.eps_end = cfg["eps_end"]
        self.eps_decay = cfg["eps_decay"]
        self.update_target_every = cfg.get("target_update", 100)
        self.step_count = 0
    
    # ε-贪婪策略进行动作选择
    # 随着训练进行，ε逐步衰减，逐步偏向利用
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.qnet.net[-1].out_features)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q = self.qnet(s)
            return q.argmax(dim=1).item()

    # 将一次状态转移（经验）保存至经验池中
    def push_transition(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def update(self):
        if len(self.buffer) < self.buffer.batch_size:
            return
        s, a, r, s2, d = self.buffer.sample()
        s  = s.to(self.device)
        a  = a.to(self.device)
        r  = r.to(self.device)
        s2 = s2.to(self.device)
        d  = d.to(self.device)

        # 当前网络计算 Q 值：
        # Q(s,a)
        qvals = self.qnet(s).gather(1, a)
        
        # 目标网络计算最大下一状态 Q 值
        # target: r + gamma * max_a' Q_target(s2, a') * (1 - done)
        with torch.no_grad():
            qnext = self.target(s2).max(dim=1, keepdim=True)[0]
            qtarget = r + self.gamma * qnext * (1 - d)

        loss = nn.functional.mse_loss(qvals, qtarget)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # 跟踪指标
        self.last_loss = loss.item()
        self.last_qval = qvals.mean().item()

        # ε 衰减
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

        # 更新网络
        self.step_count += 1
        if self.step_count % self.update_target_every == 0:
            self.target.load_state_dict(self.qnet.state_dict())
        
        return self.last_loss, self.last_qval

