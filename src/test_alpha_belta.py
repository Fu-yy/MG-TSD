import numpy as np
import matplotlib.pyplot as plt


def create_custom_beta_schedule(num_timesteps, beta_begin=1e-4,beta_end=0.1,end_ratio=0.2, decay_type="linear"):
    """
    创建 beta 调度:
    - 前 `end_ratio * num_timesteps` 使用不同的速率函数从 0 增长到 1。
    - 后续保持为 1。

    Parameters:
        num_timesteps: int, 总时间步数
        end_ratio: float, 前 end_ratio 的范围内增长到 1
        decay_type: str, 增长速率类型, 可选 'linear', 'quadratic', 'exponential'

    Returns:
        beta: np.array, 增长调度
    """
    end_step = int(num_timesteps * end_ratio)  # 前 end_ratio 时间步的长度
    beta = np.zeros(num_timesteps)  # 初始化 beta
    beta.fill(beta_end)

    # 根据不同的速率定义增长函数
    if decay_type == "linear":
        growth = np.linspace(beta_begin, beta_end, end_step)  # 线性增长
    elif decay_type == "quadratic":
        growth = np.linspace(beta_begin, beta_end, end_step) ** 2  # 平方增长
    elif decay_type == "exponential":
        growth = 1 - np.exp(-np.linspace(beta_begin, beta_end, end_step) * 5)  # 指数增长
    else:
        raise ValueError(f"Unknown decay type: {decay_type}")

    # 填充前 end_step 部分和后续部分
    beta[:end_step] = growth
    # 后续部分保持为 1（已经初始化为 1，无需重复赋值）

    return beta

# 定义 beta -> alpha 的转换
def beta_to_alpha(beta):
    """
    根据 beta 计算 alpha.
    Parameters:
        beta: np.array, beta 调度
    Returns:
        alpha: np.array, alpha 调度
    """
    return 1 - beta



# 参数设置
num_timesteps = 100  # 总时间步数
end_ratio = 0.4  # 在前 20% 衰减到 0
decay_types = ["linear"]  # 不同的下降速率类型

# 生成不同类型的 alpha 调度
betas1 = {decay: create_custom_beta_schedule(num_timesteps,end_ratio= 0.2, decay_type=decay) for decay in decay_types}
betas2 = {decay: create_custom_beta_schedule(num_timesteps, end_ratio=0.4, decay_type=decay) for decay in decay_types}
betas3 = {decay: create_custom_beta_schedule(num_timesteps, end_ratio=1, decay_type=decay) for decay in decay_types}
alphas1 = {decay: beta_to_alpha(alpha1) for decay, alpha1 in betas1.items()}
alphas2 = {decay: beta_to_alpha(alpha2) for decay, alpha2 in betas2.items()}
alphas3 = {decay: beta_to_alpha(alpha3) for decay, alpha3 in betas3.items()}

# 绘图
timesteps = np.arange(num_timesteps)
plt.figure(figsize=(12, 8))

# 绘制 alpha 和 beta
for decay in decay_types:
    plt.plot(timesteps, alphas1[decay], label=f"Alpha1 ({decay})", linestyle="--")
    plt.plot(timesteps, alphas2[decay], label=f"Alpha2 ({decay})", linestyle="--")
    plt.plot(timesteps, alphas3[decay], label=f"Alpha3 ({decay})", linestyle="--")
    plt.plot(timesteps, betas1[decay], label=f"Beta1 ({decay})", marker="o")
    plt.plot(timesteps, betas2[decay], label=f"Beta2 ({decay})", marker="o")
    plt.plot(timesteps, betas3[decay], label=f"Beta3 ({decay})", marker="o")

plt.title("Alpha and Beta Schedules with Different Decay Rates", fontsize=16)
plt.xlabel("Timesteps", fontsize=14)
plt.ylabel("Values", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()
