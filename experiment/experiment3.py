import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np


# ==========================================
# 1. 模拟环境 (Dummy Environment)
# ==========================================
class DummyEnv:
    def __init__(self):
        self.U_cpu = 0.3
        self.U_mem = 0.3
        self.latency = 20.0  # 初始延迟 20ms

        # 论文公式相关的超参数
        self.L_th = 50.0  # SLA 规定的最大延迟阈值 (ms)
        self.S_base = 0.95  # 预设的请求成功率基线 (95%)
        self.lambda_1 = 1.0  # 响应时间奖励的权重
        self.lambda_2 = 1.0  # 成功率奖励的权重
        self.max_replicas_to_evict = 3  # EA 最大可驱逐的副本数 (k=3)

    def reset(self):
        self.U_cpu = np.random.uniform(0.2, 0.4)
        self.U_mem = np.random.uniform(0.2, 0.4)
        self.latency = 20.0
        # 环境返回原始观测值 (Observation): [U_cpu, U_mem, Latency]
        # 注意: 此时 O_LSTM 还没有产生，由 Agent 的预测器计算
        return np.array([self.U_cpu, self.U_mem, self.latency / 100.0])  # 延迟除以100作归一化

    def step(self, action, actor_type):
        # --- 动作影响逻辑 (Action Space Application) ---
        if actor_type == 'DA':
            # DA 动作: 连续值 a_da 在 [0, 1] 之间，代表释放资源的比例
            # 释放资源会使得系统 CPU/Mem 利用率下降
            release_ratio = action
            self.U_cpu -= release_ratio * 0.1
            self.U_mem -= release_ratio * 0.1
        elif actor_type == 'EA':
            # EA 动作: 离散值 a_ea 在 {0, 1, 2, 3}，代表驱逐的副本数
            # 强制驱逐会大幅度释放资源，但也可能导致剧烈波动
            evict_count = action
            self.U_cpu -= evict_count * 0.15
            self.U_mem -= evict_count * 0.15

        # 模拟外部持续增加的负载和随机噪声
        self.U_cpu += 0.05 + np.random.randn() * 0.02
        self.U_mem += 0.05 + np.random.randn() * 0.02
        self.U_cpu = np.clip(self.U_cpu, 0.0, 1.0)
        self.U_mem = np.clip(self.U_mem, 0.0, 1.0)

        # --- 模拟系统性能指标 (Latency & Success Rate) ---
        mean_res_usage = (self.U_cpu + self.U_mem) / 2.0

        # 延迟 L_t: 资源利用率越高，延迟越大 (模拟冷启动等情况)
        self.latency = 20.0 + 80.0 * (mean_res_usage ** 2) + np.random.randn() * 5.0
        L_t = max(10.0, self.latency)

        # 成功率 s_t: 资源超过 70% 后，成功率开始下降
        if mean_res_usage > 0.7:
            s_t = 1.0 - 0.5 * (mean_res_usage - 0.7) + np.random.randn() * 0.02
        else:
            s_t = 1.0 - np.abs(np.random.randn() * 0.01)
        s_t = np.clip(s_t, 0.0, 1.0)

        # --- 按照论文公式计算 Reward ---
        # 公式 1: Response Time Reward (R_rt)
        if L_t <= self.L_th:
            R_rt = 1.0
        else:
            R_rt = np.exp(- ((L_t - self.L_th) / self.L_th) ** 2)

        # 公式 2: Request Success Rate Reward (R_rs)
        R_rs = np.exp((s_t - self.S_base) ** 3)

        # 总奖励 (Total Reward)
        reward = self.lambda_1 * R_rt + self.lambda_2 * R_rs

        # 判断是否崩溃 (如果资源超过95%)
        done = bool(self.U_cpu > 0.95 or self.U_mem > 0.95)
        if done:
            reward -= 5.0  # 给个崩溃惩罚

        next_obs = np.array([self.U_cpu, self.U_mem, self.latency / 100.0])
        return next_obs, reward, done


# ==========================================
# 2. 神经网络定义 (Predictor, Actors, Critic)
# ==========================================

class LSTMPredictor(nn.Module):
    def __init__(self, obs_dim=3, hidden_dim=32):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # 输出一个概率 O_LSTM

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步，经过 Sigmoid 压缩到 [0, 1] 表示概率
        prob = torch.sigmoid(self.fc(lstm_out[:, -1, :]))
        return prob


class ActorDA(nn.Module):
    """ Degradation Actor: 连续动作 a_da \in [0, 1] """

    def __init__(self, state_dim=4):
        super(ActorDA, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.mu = nn.Linear(64, 1)  # 动作的均值
        self.sigma = nn.Linear(64, 1)  # 动作的标准差

    def forward(self, state):
        x = F.relu(self.fc1(state))
        # 均值使用 Sigmoid 限制在 [0, 1] 之间
        mu = torch.sigmoid(self.mu(x))
        # 标准差必须为正
        sigma = F.softplus(self.sigma(x)) + 1e-5
        return mu, sigma


class ActorEA(nn.Module):
    """ Enforcement Actor: 离散动作 a_ea \in {0, 1, ..., k} """

    def __init__(self, state_dim=4, k_actions=4):  # k=3, 动作空间为 {0,1,2,3}
        super(ActorEA, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.logits = nn.Linear(64, k_actions)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        action_probs = F.softmax(self.logits(x), dim=-1)
        return action_probs


class Critic(nn.Module):
    def __init__(self, state_dim=4):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return self.value(x)


# ==========================================
# 3. DRL-SDS 智能体 (Agent)
# ==========================================

class DRL_SDS_Agent:
    def __init__(self, sequence_length=5):
        self.sequence_length = sequence_length
        self.gamma = 0.99
        self.threshold = 0.8  # 判断是否启用 EA 的阈值 (reward threshold 或 LSTM prob)

        # 初始化论文中定义的所有网络
        self.predictor = LSTMPredictor()
        self.actor_da = ActorDA(state_dim=4)  # State = {U_cpu, U_mem, O_LSTM, Latency}
        self.actor_ea = ActorEA(state_dim=4, k_actions=4)
        self.critic = Critic(state_dim=4)

        # 优化器
        lr = 0.001
        self.pred_opt = optim.Adam(self.predictor.parameters(), lr=lr)
        self.da_opt = optim.Adam(self.actor_da.parameters(), lr=lr)
        self.ea_opt = optim.Adam(self.actor_ea.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        self.obs_history = []

    def get_full_state(self, obs):
        """ 组合论文中定义的状态 S = {U_cpu, U_mem, O_LSTM, Latency} """
        self.obs_history.append(obs)

        # 保证历史数据长度
        if len(self.obs_history) < self.sequence_length:
            padded_history = [self.obs_history[0]] * (self.sequence_length - len(self.obs_history)) + self.obs_history
        else:
            padded_history = self.obs_history[-self.sequence_length:]

        history_tensor = torch.FloatTensor(np.array(padded_history)).unsqueeze(0)

        # 1. 获取 O_LSTM (预测概率)
        with torch.no_grad():
            O_LSTM = self.predictor(history_tensor).item()

        # 2. 拼接 State: obs 中已经包含 U_cpu, U_mem, Latency
        # 状态顺序: [U_cpu, U_mem, O_LSTM, Latency_norm]
        full_state = np.array([obs[0], obs[1], O_LSTM, obs[2]])
        return full_state, O_LSTM

    def select_action(self, full_state, O_LSTM):
        state_tensor = torch.FloatTensor(full_state).unsqueeze(0)

        # 根据 LSTM 预测结果选择 Actor (双演员选择机制)
        # 如果预测容器即将崩溃/过载 (概率很高)，则启用 EA；否则使用 DA
        if O_LSTM < self.threshold:
            # --- 使用 DA (连续动作) ---
            mu, sigma = self.actor_da(state_tensor)
            dist = Normal(mu, sigma)
            action_tensor = dist.sample()
            # 限制动作在 [0, 1] 范围内
            action = torch.clamp(action_tensor, 0.0, 1.0).item()
            log_prob = dist.log_prob(action_tensor)
            actor_name = "DA"
        else:
            # --- 使用 EA (离散动作) ---
            probs = self.actor_ea(state_tensor)
            dist = Categorical(probs)
            action_tensor = dist.sample()
            action = action_tensor.item()
            log_prob = dist.log_prob(action_tensor)
            actor_name = "EA"

        return action, log_prob, actor_name

    def update(self, state, next_state, reward, log_prob, done, actor_name):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        reward_tensor = torch.FloatTensor([reward]).unsqueeze(0)

        # Critic 评估 V(s) 和 V(s_{t+1})
        v_t = self.critic(state_tensor)
        v_next = self.critic(next_state_tensor)

        # 计算 TD Error: \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
        target = reward_tensor + self.gamma * v_next * (1 - int(done))
        td_error = (target - v_t).detach()

        # 1. 更新 Critic
        critic_loss = F.mse_loss(v_t, target.detach())
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # 2. 更新对应的 Actor
        actor_loss = -(log_prob * td_error).mean()
        if actor_name == "DA":
            self.da_opt.zero_grad()
            actor_loss.backward()
            self.da_opt.step()
        else:
            self.ea_opt.zero_grad()
            actor_loss.backward()
            self.ea_opt.step()

    def train(self, env, num_episodes=200):
        print("\n--- DRL-SDS 开始训练 ---")
        for ep in range(num_episodes):
            obs = env.reset()
            self.obs_history = []
            ep_reward = 0

            for step in range(100):
                # 组合论文定义的状态 s_t
                full_state, O_LSTM = self.get_full_state(obs)

                # 选择动作 (DA 或 EA)
                action, log_prob, actor_name = self.select_action(full_state, O_LSTM)

                # 与环境交互
                next_obs, reward, done = env.step(action, actor_name)

                # 获取 next_state 用于计算 TD error
                next_full_state, _ = self.get_full_state(next_obs)

                # 更新网络
                self.update(full_state, next_full_state, reward, log_prob, done, actor_name)

                obs = next_obs
                ep_reward += reward

                if done:
                    break

            if (ep + 1) % 10 == 0:
                print(
                    f"Episode {ep + 1:03d} | Reward: {ep_reward:.2f} | 最终Actor: {actor_name} | 延迟: {env.latency:.1f}ms")


if __name__ == "__main__":
    env = DummyEnv()
    agent = DRL_SDS_Agent(sequence_length=5)
    agent.train(env)