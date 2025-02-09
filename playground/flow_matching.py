import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# -----------------------
# 1. 准备目标分布的数据
# -----------------------

# 目标分布 p1: 100 个点，均匀分布在 [100,1000]
target_points = torch.linspace(100, 1000, steps=100)  # shape: [100]

# 用于从这100个点里采样的函数
def sample_from_p1(batch_size):
    # 随机从 target_points 中抽取 batch_size 个点（有放回）
    idx = torch.randint(0, len(target_points), size=(batch_size,))
    return target_points[idx]

# 初始分布 p0: 标准正态
def sample_from_p0(batch_size):
    return torch.randn(batch_size)


# -----------------------
# 2. 定义网络 v_\theta(t,x)
# -----------------------
class VelocityField(nn.Module):
    """
    这里用一个非常简单的 MLP:
    输入 (t, x)，输出一个标量速度 v。
    """
    def __init__(self, hidden_dim=64):
        super(VelocityField, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, t, x):
        """
        t, x 均可视为 [batch_size] 的张量
        将它们拼到一起变成 [batch_size, 2]
        """
        # 拼接输入
        inp = torch.stack([t, x], dim=1)  # shape: [batch_size, 2]
        return self.net(inp).squeeze(-1)  # 输出 shape: [batch_size]


# -----------------------
# 3. 训练设置
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VelocityField(hidden_dim=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# 超参数
num_epochs = 2000
batch_size = 128


# -----------------------
# 4. 训练循环
# -----------------------
for epoch in range(num_epochs):
    # 1) 从 p0, p1 各采一批
    x0 = sample_from_p0(batch_size).to(device)               # shape: [batch_size]
    x1 = sample_from_p1(batch_size).to(device)               # shape: [batch_size]

    # 2) 随机采一批 t \in [0,1]
    t = torch.rand(batch_size).to(device)                    # shape: [batch_size]

    # 3) 计算线性插值 x_t = (1-t)*x0 + t*x1
    x_t = (1.0 - t) * x0 + t * x1

    # 4) 计算真速度(导数): (x_1 - x_0), 在该对 (x0,x1) 下是常数
    true_velocity = x1 - x0  # shape: [batch_size]

    # 5) 网络预测速度
    pred_velocity = model(t, x_t)  # shape: [batch_size]

    # 6) 计算并回传损失
    loss = loss_fn(pred_velocity, true_velocity)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 200 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 训练结束


# -----------------------
# 5. 用训练好的模型来采样
# -----------------------
@torch.no_grad()
def generate_samples_with_flow(num_samples=10, steps=50):
    """
    使用简易的欧拉法从 t=0 积分到 t=1。
    steps 越多，数值解越精细。
    """
    # 1) 从 p0 采样初始点
    x = sample_from_p0(num_samples).to(device)  # shape [num_samples]
    t0, t1 = 0.0, 1.0
    dt = (t1 - t0) / steps

    for i in range(steps):
        t_cur = t0 + i * dt
        # 这里的 t_cur 是标量，但 x 是批量，所以要构造一个向量
        t_vec = torch.full_like(x, t_cur)
        # 计算速度
        v_cur = model(t_vec, x)
        # 欧拉步: x_{n+1} = x_n + dt * v(t_n, x_n)
        x = x + dt * v_cur
    
    # 此时 x 就是 t=1 时得到的样本
    return x.cpu().numpy()

# -----------------------
# 6. 测试采样
# -----------------------
generated = generate_samples_with_flow(num_samples=20, steps=100)
print("生成的样本：", generated)