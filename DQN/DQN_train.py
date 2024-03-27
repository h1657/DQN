import gym
from DQN import DQN, ReplayBuffer
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() \
    else torch.device("cpu")


capacity = 500  # 经验池容量
lr = 2e-3  # 学习率
gamma = 0.9  # 折扣因子
epsilon = 0.9  # 贪心系数
target_update = 200  # 目标网络的参数的更新频率
batch_size = 32
n_hidden = 128  # 隐含层神经元个数
min_size = 200  # 经验池超过200后再训练
return_list = []  # 记录每个回合的回报

# 加载环境
env = gym.make('LunarLander-v2' ,render_mode='rgb_array')
n_states = 8
n_actions = 4


# 实例化经验池
replay_buffer = ReplayBuffer(capacity)
# 实例化DQN
agent = DQN(n_states=n_states,
            n_hidden=n_hidden,
            n_actions=n_actions,
            learning_rate=lr,
            gamma=gamma,
            epsilon=epsilon,
            target_update=target_update,
            device=device,
            )

# 训练模型
return_list = []  # 用于存储每个回合的回报

for i in range(200):  # 100回合
    state = env.reset()[0]  # 每个回合开始前重置环境
    episode_return = 0  # 记录每个回合的回报
    done = False

    while not done:
        action = agent.take_action(state)  # 获取当前状态下需要采取的动作
        next_state, reward, done, _ , _ = env.step(action)  # 更新环境
        replay_buffer.add(state, action, reward, next_state, done)  # 添加经验池
        state = next_state  # 更新当前状态
        episode_return += reward  # 更新回合回报

        if replay_buffer.size() > min_size:
            s, a, r, ns, d = replay_buffer.sample(batch_size)  # 从经验池中随机抽样作为训练集
            transition_dict = {
                'states': s,
                'actions': a,
                'next_states': ns,
                'rewards': r,
                'dones': d,
            }
            agent.update(transition_dict)  # 网络更新

    return_list.append(episode_return)  # 记录每个回合的回报

    print(episode_return)

# 绘制奖励变化图
plt.plot(return_list)
plt.title('Episode Return Over Time')
plt.xlabel('Episode')
plt.ylabel('Return')
plt.show()