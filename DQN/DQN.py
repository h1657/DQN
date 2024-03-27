import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import collections
import random

class ReplayBuffer():
    def __init__(self, capacity):
        # 创建一个先进先出的队列，最大长度为capacity，保证经验池的样本量不变
        self.buffer = collections.deque(maxlen=capacity)

    # 将数据以元组形式添加进经验池
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    # 随机采样batch_size行数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)  # list, len=32

        # *transitions代表取出列表中的值，即32项
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)

class Net(nn.Module):
    def __init__(self, n_states, n_hidden, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hidden)

        # n_actions代表动作数，即输出各个动作的Q值
        self.fc2 = nn.Linear(n_hidden, n_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class DQN:
    def __init__(self, n_states, n_hidden, n_actions,
                 learning_rate, gamma, epsilon,
                 target_update, device):

        self.n_states = n_states  # 状态的特征数
        self.n_hidden = n_hidden  # 隐含层个数
        self.n_actions = n_actions  # 动作数
        self.learning_rate = learning_rate  # 训练时的学习率
        self.gamma = gamma  # 折扣因子，对下一状态的回报的缩放
        self.epsilon = epsilon  # 贪婪策略，有1-epsilon的概率探索
        self.target_update = target_update  # 目标网络的参数的更新频率
        self.device = device
        # 计数器，记录迭代次数
        self.count = 0

        # q_net用于训练参数，target_q_net用于得到下一个状态的Q值
        # 初始化两个网络相同
        self.q_net = Net(self.n_states, self.n_hidden, self.n_actions)
        self.target_q_net = Net(self.n_states, self.n_hidden, self.n_actions)

        # 优化器，更新训练网络的参数
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)

    # 选择动作
    def take_action(self, state):
        state = torch.unsqueeze(torch.from_numpy(state).float(), 0)
        # state = torch.from_numpy(state).float()
        # state = torch.Tensor(state[np.newaxis, :])

        # np.random.random()取一个0-1之间的随机数
        # 如果随机数小于epsilon就取最大的值对应的索引
        if np.random.random() < self.epsilon:

            # 获取该状态对应的动作的reward
            actions_value = self.q_net(state)

            # 获取reward最大值对应的动作索引
            action = actions_value.argmax().item()

        # 如果大于epsilon就随机探索
        else:
            # 随机选择一个动作
            action = np.random.randint(self.n_actions)
        return action

    def update(self, transition_dict):  # 传入经验池中的batch个样本
        # 获取当前时刻的状态
        states = torch.tensor(transition_dict['states'], dtype=torch.float)

        # 获取当前时刻采取的动作, 维度扩充 [b,1]
        actions = torch.tensor(transition_dict['actions']).view(-1, 1)

        # 当前状态下采取动作后得到的奖励, 维度扩充 [b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1)

        # 下一时刻的状态
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)

        # 是否终止
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1)

        # 输入当前状态，得到采取各运动得到的奖励
        # 选取已经采取的动作对应的Q值
        q_values = self.q_net(states).gather(1, actions)

        # 选出下个状态采取的动作中最大的q值
        # 因为公式里Q_pi(S_t+1 , pi(S_t+1))是pi(S_t+1)，而pi(s)就是选取最大值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)

        # 输出的当前状态的q(state_value)：即时奖励+折扣因子*下个时刻的最大回报
        # 终止状态dones=1，1-dones=0，不需要加上下一状态的价值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        # 两个网络之间的均方误差损失
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))

        self.optimizer.zero_grad()
        dqn_loss.backward()

        # 对q_net网络更新
        self.optimizer.step()

        # 在一段时间后更新target_q_net网络的参数
        if self.count % self.target_update == 0:

            # 将目标网络的参数替换成训练网络的参数
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())

        self.count += 1